import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from datetime import timedelta
import random
import itertools
import optuna
import os
import lightgbm as lgb
import gc

def completar_dataset(datos_unidos):
  """
  Completa el dataset con todas las combinaciones de SKU, STORE_ID y DATE, rellenando 
  TOTAL_SALES = 0 los días en que no hubo ventas, además de respetar las categorías de 
  cada producto y la ubicación de las tiendas.
  """

  # Fechas y combinaciones
  todas_las_fechas = pd.date_range(datos_unidos['DATE'].min(),
                                  datos_unidos['DATE'].max(),
                                  freq='D')

  todos_los_skus = datos_unidos['SKU'].unique()
  todas_las_tiendas = datos_unidos['STORE_ID'].unique()

  # Grid base
  combinaciones = pd.MultiIndex.from_product(
      [todos_los_skus, todas_las_tiendas, todas_las_fechas],
      names=['SKU', 'STORE_ID', 'DATE']
  ).to_frame(index=False)

  combinaciones['DATE'] = pd.to_datetime(combinaciones['DATE'])

  cols_extra = [
    'SKU', 'STORE_ID', 'REGION',
       'CITY', 'STATE', 'STORE_TYPE', 'OPENDATE', 'CLOSEDATE',
       'STORE_SUBGROUP_DATE_ID', 'CATEGORY', 'GROUP', 'SUBGROUP', 'GROUP_TYPE',
       'PRICE_GROUP_ID', 'BRAND'
  ]

  extra_info = (
      datos_unidos[cols_extra]
      .drop_duplicates(['SKU', 'STORE_ID'])   # 1 fila por SKU x STORE_ID
  )

  resultado = combinaciones.merge(
      extra_info,
      on=['SKU', 'STORE_ID'],
      how='left'
  )

  resultado = resultado.merge(
      datos_unidos[["SKU", "STORE_ID", "DATE", "TOTAL_SALES", "PRICE", "QUANTITY", "COSTOS"]],
      on=['SKU', 'STORE_ID', 'DATE'],
      how='left'
  )

  resultado['TOTAL_SALES'] = resultado['TOTAL_SALES'].fillna(0)
  resultado['QUANTITY'] = resultado['QUANTITY'].fillna(0)

  resultado["PRICE"] = resultado.groupby(['SKU'])["PRICE"].transform(lambda x : x.ffill()).bfill()
  resultado["COSTOS"] = resultado.groupby(['SKU'])["COSTOS"].transform(lambda x : x.ffill()).bfill()

  return resultado


def rolling_sales(datos_unidos, group, windows=[7,30,90], std=True):
    """
    Dado un dataframe con las transacciones, realiza las medias moviles de TOTAL_SALES
    agrupados por la columna group en diferentes ventanas temporales (windows)
    """

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

        if std:
            subgroup_daily[nueva_col_std] = (
            subgroup_daily.groupby(group)['TOTAL_SALES']
            .apply(lambda x: x.shift().rolling(window=7, min_periods=1).std().fillna(0)) # para el primer dia el valor es 0
            .reset_index(drop=True)
            )

        # Asignar columnas al dataframe original
        merge_cols = [group, 'DATE', nueva_col_mean]

        if std:
            merge_cols.append(nueva_col_std)

        merged = datos_unidos.merge(subgroup_daily[merge_cols], on=[group, 'DATE'], how='left')
        datos_unidos[nueva_col_mean] = merged[nueva_col_mean]

        if std:
            datos_unidos[nueva_col_std] = merged[nueva_col_std]


def rolling_price_pct(datos_unidos, group, windows=[7, 30, 90], std=True):
    """
    Calcula medias móviles de cambios porcentuales de PRICE en relación al primer precio
    registrado de cada SKU, agrupando por 'group', en distintas ventanas temporales (windows)
    """

    # Asegurar que PRICE sea numérico
    datos_unidos['PRICE'] = pd.to_numeric(datos_unidos['PRICE'], errors='coerce')

    # Calcular el cambio porcentual respecto al primer precio de cada SKU
    datos_unidos['price_pct_change'] = (
        datos_unidos.groupby('SKU')['PRICE']
        .transform(lambda x: (x - x.iloc[0]) / x.iloc[0])
        .astype(float)  # aseguramos tipo float
    )

    for window in windows:
        nueva_col_mean = f'{group}_price_pct_mean_{window}D'
        nueva_col_std = f'{group}_price_pct_std_{window}D'

        # Calcular promedio diario por grupo
        group_daily = (
            datos_unidos.groupby([group, 'DATE'], as_index=False)['price_pct_change']
            .mean()
            .sort_values([group, 'DATE'])
        )

        # Media móvil excluyendo el día actual
        group_daily[nueva_col_mean] = (
            group_daily.groupby(group)['price_pct_change']
            .apply(lambda x: x.shift().rolling(window, min_periods=1).mean().fillna(0))
            .reset_index(drop=True)
        )

        if std:
            group_daily[nueva_col_std] = (
                group_daily.groupby(group)['price_pct_change']
                .apply(lambda x: x.shift().rolling(window, min_periods=1).std().fillna(0))
                .reset_index(drop=True)
            )

        # Merge al dataframe original
        merge_cols = [group, 'DATE', nueva_col_mean]
        if std:
            merge_cols.append(nueva_col_std)

        datos_unidos = datos_unidos.merge(group_daily[merge_cols], on=[group, 'DATE'], how='left')
    datos_unidos.drop(columns=["price_pct_change"], inplace=True)

    return datos_unidos


def rolling_sales_completo(datos_unidos):
    """
    Dado un dataframe con las transacciones, rellena los dias que no hubo ventas para cada combinacion de
    SKU X STORE_ID con TOTAL_SALES = 0, calcula las medias moviles, las agrega al dataframe original y
    lo devuelve (sin completar)
    """

    # Guardamos el índice donde cambia la fecha para acceso rápido
    cambios_dia = datos_unidos["DATE"].ne(datos_unidos["DATE"].shift()).to_numpy().nonzero()[0]
    fechas_unicas = datos_unidos["DATE"].unique()

    # Lista única de combinaciones SKU-STORE_ID
    combinaciones = datos_unidos[["SKU", "STORE_ID"]].drop_duplicates()

    def rellenar_faltantes(df, fecha):
        # Todas las combinaciones para esta fecha
        comb_fecha = combinaciones.copy()
        comb_fecha["DATE"] = fecha
        # Merge para meter TOTAL_SALES=0 donde falta
        df_completo = comb_fecha.merge(df, on=["SKU", "STORE_ID", "DATE"], how="left")
        df_completo["TOTAL_SALES"] = df_completo["TOTAL_SALES"].fillna(0)
        return df_completo

    buffer = pd.DataFrame()
    resultados = []
    windows = [7, 30, 90]

    for window in windows:
        datos_unidos[f"SKU_STORE_mean_{window}D"] = pd.NA

        for fecha in fechas_unicas:
            # Datos del día actual
            df_dia = datos_unidos.loc[datos_unidos["DATE"] == fecha, ["SKU", "STORE_ID", "DATE", "TOTAL_SALES"]]
            df_dia_completo = rellenar_faltantes(df_dia, fecha)

            # Agregar al buffer
            buffer = pd.concat([buffer, df_dia_completo], ignore_index=True)

            # Mantener sólo los últimos window+1 días (para limitar memoria)
            if buffer["DATE"].nunique() > window+1:
                fecha_mas_vieja = buffer["DATE"].min()
                buffer = buffer[buffer["DATE"] != fecha_mas_vieja]

            # Filas originales del día actual
            df_original_dia = datos_unidos.loc[datos_unidos["DATE"] == fecha,
                                            ["SKU", "STORE_ID", "DATE", "TOTAL_SALES"]]

            # Calcular promedio con los días previos que haya
            dias_previos = sorted(buffer["DATE"].unique())[:-1]  # todos menos el actual

            if len(dias_previos) > 0:
                # Tomar como máximo window días previos
                dias_a_usar = dias_previos[-window:]
                df_prev = buffer[buffer["DATE"].isin(dias_a_usar)]
                media_prev = df_prev.groupby(["SKU", "STORE_ID"], observed=False)["TOTAL_SALES"].mean().reset_index()
                media_prev["DATE"] = fecha
                media_prev.rename(columns={"TOTAL_SALES": f"SKU_STORE_mean_{window}D"}, inplace=True)

                # Actualizar directamente en el dataset original
                idx_update = datos_unidos.index[datos_unidos["DATE"] == fecha]
                merged = datos_unidos.loc[idx_update, ["SKU", "STORE_ID", "DATE"]].merge(
                    media_prev, on=["SKU", "STORE_ID", "DATE"], how="left")

                # Si no se creó la columna en el merge, la creamos con NaN
                if f"SKU_STORE_mean_{window}D" not in merged.columns:
                    merged[f"SKU_STORE_mean_{window}D"] = pd.NA

                datos_unidos.loc[idx_update, f"SKU_STORE_mean_{window}D"] = merged[f"SKU_STORE_mean_{window}D"].values

        datos_unidos.fillna({f"SKU_STORE_mean_{window}D":0}, inplace=True)

    return datos_unidos


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


def walk_forward_lightgbm(df, features, target_col, date_col, categorical_cols,
                              train_days=365, step_days=30, forecast_days=7,
                              params=None):
    """
    Realiza un walk-forward con LightGBM, entrenando el modelo primero con una ventana expandible (step_days) 
    y prediciendo los próximos días (forecast_days). Ademas, empieza a entrenar con un año de datos (train_days)

    df: DataFrame con features + target
    target_col: nombre de la columna objetivo ('TOTAL_SALES')
    date_col: columna con la fecha
    categorical_cols: lista de columnas categóricas (deben ser dtype 'category')
    train_days, step_days, forecast_days: enteros en días
    params: dict de parámetros LightGBM
    """

    df = df.sort_values(date_col).reset_index(drop=False)  # preservamos el índice original
    df[date_col] = pd.to_datetime(df[date_col])

    results = []
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    start_train_end = min_date + timedelta(days=train_days)

    count = 0

    while start_train_end + timedelta(days=forecast_days) <= max_date:
        count += 1
        print(f"Walk-forward: iteración número {count}")

        # Train y Test: usamos índices en lugar de sub-dataframes
        train_idx = df.index[df[date_col] < start_train_end]
        test_idx = df.index[(df[date_col] >= start_train_end) &
                            (df[date_col] < start_train_end + timedelta(days=forecast_days))]

        if len(test_idx) == 0:
            break

        # Validation set (últimos 7 días dentro de train)
        valid_days_inner = 7
        train_end_inner_date = df.loc[train_idx, date_col].max() - timedelta(days=valid_days_inner)

        train_inner_idx = train_idx[df.loc[train_idx, date_col] <= train_end_inner_date]
        valid_inner_idx = train_idx[df.loc[train_idx, date_col] > train_end_inner_date]

        # Extraemos solo cuando es necesario
        X_train_inner = df.loc[train_inner_idx, features]
        y_train_inner = df.loc[train_inner_idx, target_col]
        X_valid_inner = df.loc[valid_inner_idx, features]
        y_valid_inner = df.loc[valid_inner_idx, target_col]

        # Dataset LightGBM
        lgb_train = lgb.Dataset(X_train_inner, label=y_train_inner, categorical_feature=categorical_cols)
        lgb_valid = lgb.Dataset(X_valid_inner, label=y_valid_inner, categorical_feature=categorical_cols, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train_inner", "valid_inner"]
        )

        # Predicciones (usando índices)
        y_test_pred = model.predict(df.loc[test_idx, features], num_iteration=model.best_iteration)
        y_train_pred = model.predict(X_train_inner, num_iteration=model.best_iteration)

        # Métricas
        r2_test = r2_score(df.loc[test_idx, target_col], y_test_pred)
        r2_train = r2_score(y_train_inner, y_train_pred)

        results.append({
            "train_end_date": start_train_end,
            "r2_train": r2_train,
            "r2_test": r2_test
        })

        # Avanzamos la ventana
        start_train_end += timedelta(days=step_days)

        del X_train_inner, X_valid_inner, y_train_inner, y_valid_inner
        gc.collect()

    return pd.DataFrame(results)


def crear_price_grid(datos_unidos: pd.DataFrame, n_prices: int = 50):
    """
    Devuelve dict { SKU: array de precios posibles} usando min/max histórico por SKU.
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


def crear_price_grid_descuento(datos_unidos: pd.DataFrame, n_prices: int = 50, descuento = 0.3) -> dict:
    """
    Devuelve dict { SKU: array de precios posibles }
    usando el último precio por SKU ±30%.
    """
    # Tomamos el último precio por SKU según la fecha
    ultimos_precios = datos_unidos.sort_values("DATE").groupby("SKU")["PRICE"].last().reset_index()

    price_grid = {}
    for _, r in ultimos_precios.iterrows():
        sku = r["SKU"]
        last_price = r["PRICE"]

        if pd.isna(last_price) or last_price <= 0:
            continue  # Evitamos precios nulos o negativos

        min_p = last_price * (1-descuento)  # -30%
        max_p = last_price * (1+descuento) # +30%

        price_grid[sku] = np.linspace(min_p, max_p, n_prices)

    return price_grid


def crear_template(df, columnas_extraidas, cols_categoricas):
    """
    Dado un dataframe con las transacciones, crea un dataframe template con todas las combinaciones de SKU X STORE_ID de los
    proximos 7 dias
    """

    # Creamos un dataframe con todas las combinaciones de SKU X STORE_ID
    template = df[columnas_extraidas].drop_duplicates().reset_index(drop=True)

    # Quitamos las tiendas que ya cerraron
    # Hay 150 (numero de tiendas) . 854 (numero de sku) combinaciones
    template = template[template["YEAR_CLOSE"] > 2023]

    # Agregamos los ultimos costos de los productos
    ultimos_costos = (
        df
        .groupby(["SKU", "STORE_ID"], as_index=False)
        .last()[["SKU", "STORE_ID", "COSTOS"]]
    )
    template = template.merge(ultimos_costos, on=["SKU", "STORE_ID"], how="left")

    # Cada uno de los 7 dias tendra todas las combinaciones
    fechas = pd.date_range(start="2024-01-01", periods=7, freq="D")
    df_fechas = pd.DataFrame({"DATE": fechas})

    template = (
        df_fechas.assign(key=1)
        .merge(template.assign(key=1), on="key")
        .drop(columns="key")
    )

    # Features agregados
    template["DATE"] = pd.to_datetime(template["DATE"])
    template["YEAR"] = template["DATE"].dt.year
    template["MONTH"] = template["DATE"].dt.month
    template["DAY"] = template["DATE"].dt.day
    template["DAY_OF_WEEK"] = template["DATE"].dt.day_name()
    template["WEEK"] = template["DATE"].dt.isocalendar().week

    # Pasamos las columnas al type adecaudo
    for col in cols_categoricas:
        template[col] = template[col].astype("category")

    return template


def rolling_sales_template(df, template, group, windows=[30, 90, 180], std=True):
    """
    Dado los datos copmletos df y un template con las posibles transacciones de la siguiente semana,
    calcula los rolling features de TOTAL_SALES agrupados por group en distintas ventanas temporales (windows)
    """

    for window in windows:
        df[f"tem_{group}_mean_{window}D"] = (
            df
            .groupby(group)["TOTAL_SALES"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        # Obtener solo la última fila por group
        ultimos_promedios = (
            df
            .sort_values("DATE")
            .groupby(group)
            .tail(1)[[group, f"tem_{group}_mean_{window}D"]]
        )

        # Renombrar la columna para el merge
        ultimos_promedios = ultimos_promedios.rename(
            columns={f"tem_{group}_mean_{window}D": f"{group}_mean_{window}D"}
        )

        # Hacer el merge con template
        template = template.merge(
            ultimos_promedios,
            on=group,
            how="left"
        )

        # Eliminar las columnas temporales de df
        df.drop(columns=[f"tem_{group}_mean_{window}D"], inplace=True)

        # Hacemos lo mismo con std
        if std:
            df[f"tem_{group}_std_{window}D"] = (
                df
                .groupby(group)["TOTAL_SALES"]
                .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
            )

            ultimos_std = ultimos_std.rename(
                columns={f"tem_{group}_std_{window}D": f"{group}_std_{window}D"}
            )

            template = template.merge(
            ultimos_std,
            on=group,
            how="left"
            )

            ultimos_std = (
                df
                .sort_values("DATE")
                .groupby(group)
                .tail(1)[[group, f"tem_{group}_std_{window}D"]]
            )

            df.drop(columns=[ f"tem_{group}_std_{window}D"], inplace=True)

    return template


def rolling_price_template(df, template, group, windows=[30,90], std=True):
    """
    Agrega al template el promedio de los cambios porcentuales de precio (vs. primer precio del SKU)
    de los últimos 7 días, calculado por 'group'.

    Parámetros:
    df : pd.DataFrame
        Debe contener: DATE, SKU, PRICE y la columna 'group' (ej. STORE_ID, SUBGROUP).
    template : pd.DataFrame
        Combinaciones futuras (ej. SKU x STORE_ID) donde se agregará la columna.
    group : str
        Nombre de la columna por la cual agrupar (ej. 'STORE_ID', 'SUBGROUP').

    Retorna:
    pd.DataFrame
        template con la nueva columna: f"{group}_avg_price_change_last7"
    """

    df['tem_price_pct_change'] = (
        df.groupby('SKU')['PRICE']
        .transform(lambda x: (x - x.iloc[0]) / x.iloc[0])
        .astype(float)  # aseguramos tipo float
    )

    # Filtramos las ultimas fechas
    df["DATE"] = pd.to_datetime(df["DATE"])
    end_date = df['DATE'].max()
    start_date = pd.Timestamp(end_date) - pd.Timedelta(days=max(windows))
    df_ult = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)].copy()

    # Eliminar las columnas temporales de df
    df.drop(columns=["tem_price_pct_change"], inplace=True)

    for window in windows:
        df_ult[f"tem_{group}_mean_{window}D"] = (
                df_ult
                .groupby(group)["tem_price_pct_change"]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

        # Obtener solo la última fila por group
        ultimos_promedios = (
            df_ult
            .sort_values("DATE")
            .groupby(group)
            .tail(1)[[group, f"tem_{group}_mean_{window}D"]]
        )

        # Renombrar la columna para el merge
        ultimos_promedios = ultimos_promedios.rename(
            columns={f"tem_{group}_mean_{window}D": f'{group}_price_pct_mean_{window}D'}
        )

        # Hacer el merge con template
        template = template.merge(
            ultimos_promedios,
            on=group,
            how="left"
        )

        # Hacemos lo mismo con std
        if std:
            df_ult[f"tem_{group}_std_{window}D"] = (
                df_ult
                .groupby(group)["tem_price_pct_change"]
                .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
            )

            ultimos_std = (
                df_ult
                .sort_values("DATE")
                .groupby(group)
                .tail(1)[[group, f"tem_{group}_std_{window}D"]]
            )

            ultimos_std = ultimos_std.rename(
                columns={f"tem_{group}_std_{window}D": f'{group}_price_pct_std_{window}D'}
            )

            template = template.merge(
            ultimos_std,
            on=group,
            how="left"
            )

    return template


def crear_csv_kaggle(df_best, dummy_subgroup='DUMMY'):
    """
    Construye un dataframe con todas las combinaciones STORE_ID x SUBGROUP x DATE,
    incluyendo un subgrupo dummy con TOTAL_SALES=0, y suma TOTAL_SALES de df_best.

    Parámetros:
    -----------
    df_best : pd.DataFrame
        Debe contener columnas: DATE, STORE_ID, SUBGROUP, TOTAL_SALES
    dummy_subgroup : str
        Nombre del subgrupo que siempre tendrá TOTAL_SALES=0

    Retorna:
    --------
    df_result : pd.DataFrame
        Columnas: STORE_SUBGROUP_DATE_ID, TOTAL_SALES
    """

    # Asegurar formato de DATE
    df_best = df_best.copy()
    df_best['DATE'] = pd.to_datetime(df_best['DATE'])

    # Valores únicos de stores y subgrupos (agregamos dummy)
    stores = df_best['STORE_ID'].unique()
    subgroups = df_best['SUBGROUP'].unique().tolist() + [dummy_subgroup]
    dates = df_best['DATE'].unique()

    # Crear todas las combinaciones posibles
    all_combinations = pd.DataFrame(
        list(itertools.product(stores, subgroups, dates)),
        columns=['STORE_ID','SUBGROUP','DATE']
    )

    # Agrupar df_best por STORE_ID, SUBGROUP, DATE y sumar TOTAL_SALES
    df_agg = (
        df_best.groupby(['STORE_ID','SUBGROUP','DATE'], as_index=False)['TOTAL_SALES']
        .sum()
    )

    # Merge con todas las combinaciones
    df_result = all_combinations.merge(df_agg, on=['STORE_ID','SUBGROUP','DATE'], how='left')

    # Rellenar faltantes con 0 (incluye automáticamente dummy)
    df_result['TOTAL_SALES'] = df_result['TOTAL_SALES'].fillna(0)

    # Crear STORE_SUBGROUP_DATE_ID
    df_result['STORE_SUBGROUP_DATE_ID'] = (
        df_result['STORE_ID'].astype(str) + '_' +
        df_result['SUBGROUP'].astype(str) + '_' +
        df_result['DATE'].astype(str)
    )

    # Dejar solo las columnas finales
    df_result = df_result[['STORE_SUBGROUP_DATE_ID','TOTAL_SALES']]

    return df_result


def optimizacion_precios_optuna(template, model, price_grid, features, target="GAIN",
                                predict_func=None, save_dir=None, file_name="mejor_config",
                                n_trials=1000, n_jobs=-1, seed=42):
    """
    Optimiza la configuración de precios usando Optuna para maximizar GAIN o TOTAL_SALES,
    eligiendo un único precio por SKU.
    """
    np.random.seed(seed)

    # Lista de SKUs únicos
    skus_unicos = template["SKU"].unique()
    n_skus = len(skus_unicos)

    # Precalcular estructuras
    costos = template["COSTOS"].values
    precios_dict = {sku: np.array(price_grid[sku]) for sku in skus_unicos}

    # Mapear SKU -> posiciones en template (para asignar rápido)
    sku_posiciones = {sku: np.where(template["SKU"].values == sku)[0] for sku in skus_unicos}

    X_temp = template.copy()

    # Función objetivo para Optuna
    def objective(trial):
        precios_asignados = np.zeros(template.shape[0], dtype=float)

        # Elegir un precio por SKU y asignarlo a todas sus posiciones
        for sku in skus_unicos:
            opciones_precio = precios_dict[sku]
            idx = trial.suggest_int(f"sku_{sku}", 0, len(opciones_precio) - 1)
            precios_asignados[sku_posiciones[sku]] = opciones_precio[idx]

        # Predicción
        X_temp["PRICE"] = precios_asignados

        if predict_func is None:
            y_pred = model.predict(X_temp[features])
        else:
            y_pred = predict_func(model, X_temp[features])

        # Cálculos de métricas
        gain = y_pred.sum() - ((y_pred / precios_asignados) * costos).sum()
        total_sales = y_pred.sum()

        if target == "GAIN":
            return gain
        elif target == "TOTAL_SALES":
            return total_sales
        else:
            raise ValueError("El parámetro target debe ser 'GAIN' o 'TOTAL_SALES'.")

    # Crear estudio Optuna
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    # Obtener mejor configuración
    best_params = study.best_params
    precios_finales = np.zeros(template.shape[0], dtype=float)
    for sku in skus_unicos:
        precio = precios_dict[sku][best_params[f"sku_{sku}"]]
        precios_finales[sku_posiciones[sku]] = precio

    # Predecir con la mejor configuración
    X_best = template.copy()
    X_best["PRICE"] = precios_finales
    if predict_func is None:
        y_best = model.predict(X_best[features])
    else:
        y_best = predict_func(model, X_best[features])

    X_best["TOTAL_SALES"] = y_best

    mejor_gain = y_best.sum() - ((y_best / precios_finales) * costos).sum()
    mejor_sales = y_best.sum()

    # Guardar si es necesario
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df_mejor = X_best.copy()
        df_mejor["TOTAL_SALES"] = y_best
        df_mejor.to_csv(f"{save_dir}/{file_name}.csv", index=False)

    return X_best, y_best, mejor_sales, mejor_gain, precios_finales, study


def optimizacion_precios_region_optuna(template, model, price_grid, features, target="GAIN",
                                predict_func=None, save_dir=None, file_name="mejor_config",
                                n_trials=1000, n_jobs=-1, seed=42):
    """
    Optimiza la configuración de precios usando Optuna para maximizar GAIN o TOTAL_SALES,
    eligiendo un precio por SKU y REGION.
    """
    np.random.seed(seed)

    # Combinaciones únicas SKU-REGION
    sku_region_pairs = template[["SKU", "REGION"]].drop_duplicates().values
    n_pairs = len(sku_region_pairs)

    # Precalcular estructuras
    costos = template["COSTOS"].values
    precios_dict = {sku: np.array(price_grid[sku]) for sku in template["SKU"].unique()}

    # Mapear (SKU, REGION) a posiciones en template (para asignar rápido)
    pair_posiciones = {
        (sku, region): np.where((template["SKU"].values == sku) & (template["REGION"].values == region))[0]
        for sku, region in sku_region_pairs
    }

    X_temp = template.copy()

    # Función objetivo para Optuna
    def objective(trial):
        precios_asignados = np.zeros(template.shape[0], dtype=float)

        # Elegir un precio por cada par (SKU, REGION)
        for sku, region in sku_region_pairs:
            opciones_precio = precios_dict[sku]
            idx = trial.suggest_int(f"sku_{sku}_region_{region}", 0, len(opciones_precio) - 1)
            precios_asignados[pair_posiciones[(sku, region)]] = opciones_precio[idx]

        # Predicción
        X_temp["PRICE"] = precios_asignados

        if predict_func is None:
            y_pred = model.predict(X_temp[features])
        else:
            y_pred = predict_func(model, X_temp[features])

        # Cálculos de métricas
        gain = y_pred.sum() - ((y_pred / precios_asignados) * costos).sum()
        total_sales = y_pred.sum()

        if target == "GAIN":
            return gain
        elif target == "TOTAL_SALES":
            return total_sales
        else:
            raise ValueError("El parámetro target debe ser 'GAIN' o 'TOTAL_SALES'.")

    # Crear estudio Optuna
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    # Obtener mejor configuración
    best_params = study.best_params
    precios_finales = np.zeros(template.shape[0], dtype=float)
    for sku, region in sku_region_pairs:
        precio = precios_dict[sku][best_params[f"sku_{sku}_region_{region}"]]
        precios_finales[pair_posiciones[(sku, region)]] = precio

    # Predecir con la mejor configuración
    X_best = template.copy()
    X_best["PRICE"] = precios_finales
    if predict_func is None:
        y_best = model.predict(X_best[features])
    else:
        y_best = predict_func(model, X_best[features])

    X_best["TOTAL_SALES"] = y_best

    mejor_gain = y_best.sum() - ((y_best / precios_finales) * costos).sum()
    mejor_sales = y_best.sum()

    # Guardar si es necesario
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df_mejor = X_best.copy()
        df_mejor["TOTAL_SALES"] = y_best
        df_mejor.to_csv(f"{save_dir}/{file_name}.csv", index=False)

    return X_best, y_best, mejor_sales, mejor_gain, precios_finales, study
