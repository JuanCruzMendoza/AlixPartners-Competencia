import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import timedelta
import random


def features_rolling_mean_std(datos_unidos, group, windows=[7,30,90]):
    for window in windows:
        nueva_col_mean = f'{group}_mean_{window}D'
        nueva_col_std = f'{group}_std_{window}D'

        subgroup_daily = (
            datos_unidos.groupby([group, 'DATE'], as_index=False)['TOTAL_SALES']
            .sum()
            .sort_values([group, 'DATE'])
        )

        # Calcular promedio móvil excluyendo el día actual
        # Shift para no incluir el valor del día actual
        subgroup_daily[nueva_col_mean] = (
            subgroup_daily.groupby(group)['TOTAL_SALES']
            .apply(lambda x: x.shift().rolling(window, min_periods=1).mean().fillna(0)) # para el primer dia el valor es 0
            .reset_index(drop=True)
        )

        subgroup_daily[nueva_col_std] = (
        subgroup_daily.groupby(group)['TOTAL_SALES']
        .apply(lambda x: x.shift().rolling(window=7, min_periods=1).std().fillna(0)) # para el primer dia el valor es 0
        .reset_index(drop=True)
        )

        # Asignar columnas al dataframe original
        merge_cols = [group, 'DATE', nueva_col_mean, nueva_col_std]
        merged = datos_unidos.merge(subgroup_daily[merge_cols], on=[group, 'DATE'], how='left')

        datos_unidos[nueva_col_mean] = merged[nueva_col_mean]
        datos_unidos[nueva_col_std] = merged[nueva_col_std]




def walk_forward_forecast(df, model, features, target, train_days=365, step_days=30, forecast_days=7):
    """
    Realiza un walk-forward, entrenando el modelo con una expanding window y prediciendo los 
    proximos dias

    df: DataFrame 
    model: modelo sklearn
    train_days: tamaño inicial del set de entrenamiento en días
    step_days: cuántos días se suman en cada iteración
    forecast_days: horizonte de predicción en días
    """
    # Aseguramos orden por fecha
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")
    
    results = []
    min_date = df["DATE"].min()
    max_date = df["DATE"].max()

    start_train_end = min_date + timedelta(days=train_days)

    while start_train_end + timedelta(days=forecast_days) <= max_date:
        # Definir ventanas
        train_data = df[df["DATE"] < start_train_end]
        test_data = df[(df["DATE"] >= start_train_end) &
                       (df["DATE"] < start_train_end + timedelta(days=forecast_days))]

        if len(test_data) == 0:
            break

        # Features y target
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        # Entrenar
        model.fit(X_train, y_train)

        # Predicciones y métricas
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        results.append({
            "train_end_date": start_train_end,
            "r2_train": r2_train,
            "r2_test": r2_test
        })

        # Avanzar ventana
        start_train_end += timedelta(days=step_days)

    return pd.DataFrame(results)


def crear_price_grid(datos_unidos: pd.DataFrame, n_prices: int = 50):
    """
    Devuelve dict { SKU: array_de_50_precios } usando min/max histórico por SKU.
    """
    price_ranges = datos_unidos.groupby('SKU')['PRICE'].agg(['min', 'max']).reset_index()
    price_grid = {}
    for _, r in price_ranges.iterrows():
        sku = r['SKU']
        min_p, max_p = r['min'], r['max']
        if pd.isna(min_p) or pd.isna(max_p):
            continue
        # Si min == max, np.linspace devuelve 1 valor repetido; está OK.
        price_grid[sku] = np.linspace(min_p, max_p, n_prices)
    return price_grid


def optimizacion_precios(template, model, price_grid, features, n_iter=1000,
                         target="GAIN", predict_func=None, save_dir=None, file_name = "mejor_config"):

    """
    Toma un Dataframe base (template) con todas las combinaciones de SKU X STORE_ID de los proximos dias y un modelo
    con su funcion predict, construye una posible configuracion de precios segun price_grid (un precio por cada SKU X STORE_ID),
    y busca la configuracion que maximiza target

    template: pandas Dataframe (con features y "COSTOS")
    model: modelo de prediccion 
    price_grid: diccionario con SKU como claves y rango de precios como valores (de la funcion crear_price_grid())
    features: lista de features que utiliza el modelo
    cant_precios: cantidad de precios que hay en cada rango
    n_iter: int
    target: "GAIN" si se maximiza la ganancia neta o "TOTAL_SALES" si se maximiza las ventas totales
    predict_func: en caso de que el modelo necesite preprocesamiento, esta la opcion de pasar una funcion especial (sino se utiliza model.predict())
    save_dir: directory para guardar los resultados
    """

    # Precalculos
    n_rows = template.shape[0]
    costos = template["COSTOS"].values
    sku_array = template["SKU"].values
    cant_precios = len(next(iter(price_grid)))

    # Mapeamos precios posibles para cada SKU en forma de matriz
    precios_dict = {sku: np.array(price_grid[sku]) for sku in price_grid}

    precios_matrix = np.zeros((n_rows, cant_precios), dtype=float)
    sku_indices = np.zeros(n_rows, dtype=int)

    for i, sku in enumerate(sku_array):
        precios_sku = precios_dict[sku]
        precios_matrix[i, :len(precios_sku)] = precios_sku
        sku_indices[i] = len(precios_sku)

    # DataFrame base sin PRICE
    X_base = template.copy()

    mejor_gain = -np.inf
    mejor_sales = -np.inf
    mejor_config = None
    mejor_y_pred = None

    for n in range(n_iter):
        print(f"Iteracion: {n+1}")
        # Elección vectorizada de precios aleatorios
        idx_random = (np.random.random(n_rows) * sku_indices).astype(int)
        precios_asignados = precios_matrix[np.arange(n_rows), idx_random]

        # Predicción
        X_base["PRICE"] = precios_asignados
        if predict_func is None:
            y_pred = model.predict(X_base[features])
        else:
            y_pred = predict_func(model, X_base[features])

        # Cálculo de ganancia
        gain = y_pred.sum() - ((y_pred / precios_asignados) * costos).sum()
        total_sales = y_pred.sum()

        # Guardar mejor configuración
        if target == "GAIN":
            if gain > mejor_gain:
                mejor_gain = gain
                mejor_sales = total_sales
                mejor_config = precios_asignados.copy()
                mejor_y_pred = y_pred

        if target == "TOTAL_SALES":
            if total_sales > mejor_sales:
                mejor_gain = gain
                mejor_sales = total_sales
                mejor_config = precios_asignados.copy()
                mejor_y_pred = y_pred

        print(f"Mejor Total sales: {mejor_sales} \nMejor gain: {mejor_gain} \n")

    # Guardar mejor configuración en memoria
    if save_dir:
        df_mejor = template.copy()
        df_mejor["PRICE"] = mejor_config
        df_mejor["TOTAL_SALES"] = mejor_y_pred
        df_mejor.to_csv(f"{save_dir}/{file_name}.csv", index=False)

    return mejor_y_pred, mejor_sales, mejor_gain, mejor_config


def features_rolling_template(df, template, group, windows=[30, 90, 180]):
    
    for window in windows:
        df[f"tem_{group}_mean_{window}D"] = (
            df
            .groupby(group)["TOTAL_SALES"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        df[f"tem_{group}_std_{window}D"] = (
            df
            .groupby(group)["TOTAL_SALES"]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )

        # Obtener solo la última fila por group
        ultimos_promedios = (
            df
            .sort_values("DATE")
            .groupby(group)
            .tail(1)[[group, f"tem_{group}_mean_{window}D"]]
        )

        ultimos_std = (
            df
            .sort_values("DATE")
            .groupby(group)
            .tail(1)[[group, f"tem_{group}_std_{window}D"]]
        )

        # Renombrar la columna para el merge
        ultimos_promedios = ultimos_promedios.rename(
            columns={f"tem_{group}_mean_{window}D": f"{group}_mean_{window}D"}
        )

        ultimos_std = ultimos_std.rename(
            columns={f"tem_{group}_std_{window}D": f"{group}_std_{window}D"}
        )

        # Hacer el merge con template
        template = template.merge(
            ultimos_promedios,
            on=group,
            how="left"
        )

        template = template.merge(
            ultimos_std,
            on=group,
            how="left"
        )

        # Eliminar las columnas temporales de df
        df = df.drop(columns=[f"tem_{group}_mean_{window}D", f"tem_{group}_std_{window}D"])

    return template