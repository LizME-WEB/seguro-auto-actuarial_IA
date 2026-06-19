# ============================================================
# utils/preprocessing.py
# Paso 3 — Preprocesamiento con Pipeline y ColumnTransformer
# Proyecto: Sistema actuarial de predicción de riesgo
# ============================================================

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,
                                   OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ── STEP 1: DEFINICIÓN DE COLUMNAS ──────────────────────────────────────────

# Columnas que NUNCA entran como features
EXCLUIR_SIEMPRE = ['poliza_id', 'clase_costo']

# Targets
TARGET_REG = 'costo_esperado_anual_mxn'
TARGET_CLF = 'riesgo_alto'

# Columnas numéricas originales (con faltantes manejados por SimpleImputer)
NUMERIC_COLS = [
    'edad_conductor',
    'edad2',                    # engineerada (edad²)
    'antiguedad_cliente_anios',
    'ingreso_mensual_mxn',
    'log_ingreso',              # engineerada (log del ingreso)
    'score_crediticio',
    'prima_mensual_mxn',
    'suma_asegurada_mxn',
    'deducible_pct',
    'historial_siniestros_3_anios',
    'km_anuales',
    'km_por_vehiculo',          # engineerada (km / (edad_vehiculo + 1))
    'edad_vehiculo_anios',
    'dias_hasta_renovacion',
    'puntaje_riesgo_zona',
    'numero_siniestros_12m',
    'ratio_prima_suma',         # engineerada (prima / suma_asegurada * 1000)
]

# Columnas categóricas nominales (sin orden → One-Hot Encoding)
NOMINAL_COLS = [
    'sexo',
    'estado_civil',
    'ocupacion',          # tiene faltantes → SimpleImputer(moda)
    'zona_residencia',
    'region',
    'tipo_vehiculo',
    'uso_vehiculo',
    'metodo_pago',
    'canal_venta',
]

# Columnas categóricas ordinales (con orden → OrdinalEncoder)
ORDINAL_COLS = ['nivel_estudios', 'segmento_marca']

# Orden explícito para OrdinalEncoder
# Crítico: las categorías deben coincidir exactamente con los valores del CSV
ORDINAL_CATEGORIES = [
    ['Secundaria', 'Preparatoria', 'Licenciatura', 'Posgrado'],  # nivel_estudios
    ['Economico', 'Medio', 'Premium'],                            # segmento_marca
]

# Columnas binarias (mapeo manual, no entran al ColumnTransformer)
BINARY_COLS = ['tiene_gps', 'asistencia_vial', 'mantenimiento_al_dia']


# ── STEP 2: FEATURE ENGINEERING ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas features derivadas antes del split train/test.
    Estas transformaciones son deterministas (no aprenden del training set)
    por lo que es seguro aplicarlas antes del split.
    """
    df = df.copy()

    # 1. log(ingreso): corrige el sesgo extremo (skew 5.67 → 0.31)
    #    np.log1p(x) = log(x + 1), maneja x=0 sin error
    df['log_ingreso'] = np.log1p(df['ingreso_mensual_mxn'])

    # 2. edad²: captura el efecto no lineal (curva en U) de la edad
    df['edad2'] = df['edad_conductor'] ** 2

    # 3. km_por_vehiculo: intensidad de uso relativa al desgaste del vehículo
    #    +1 en denominador evita división por cero (vehículos nuevos con 0 años)
    df['km_por_vehiculo'] = df['km_anuales'] / (df['edad_vehiculo_anios'] + 1)

    # 4. ratio_prima_suma: cuántos pesos paga por cada $1,000 asegurados
    #    Mide si la póliza está bien tarificada (ratio bajo = posible sub-tarificación)
    df['ratio_prima_suma'] = df['prima_mensual_mxn'] / df['suma_asegurada_mxn'] * 1000

    return df


def codificar_binarias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapea las columnas binarias de texto (Si/No) a 0/1.
    mantenimiento_al_dia: NaN → -1 (categoría informativa propia).
    """
    df = df.copy()

    df['tiene_gps'] = df['tiene_gps'].map({'Si': 1, 'No': 0}).astype(float)
    df['asistencia_vial'] = df['asistencia_vial'].map({'Si': 1, 'No': 0}).astype(float)

    # NaN en mantenimiento_al_dia → -1 (no imputar: es información por sí misma)
    df['mantenimiento_al_dia'] = (
        df['mantenimiento_al_dia']
        .map({'Si': 1, 'No': 0})
        .fillna(-1)
        .astype(float)
    )

    return df


# ── STEP 3: PREPARAR EL DATAFRAME ────────────────────────────────────────────

def preparar_dataframe(df: pd.DataFrame,
                       target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Aplica feature engineering, codifica binarias y separa X / y.

    Args:
        df: DataFrame original cargado del CSV
        target: 'costo_esperado_anual_mxn' o 'riesgo_alto'

    Returns:
        X: DataFrame de features
        y: Serie del target
    """
    df = feature_engineering(df)
    df = codificar_binarias(df)

    # Columnas a excluir: identificador + clase derivada + el otro target
    otro_target = TARGET_CLF if target == TARGET_REG else TARGET_REG
    cols_excluir = EXCLUIR_SIEMPRE + [target, otro_target]

    X = df.drop(columns=cols_excluir)
    y = df[target]

    return X, y


# ── STEP 4: CONSTRUIR EL PREPROCESADOR ───────────────────────────────────────

def construir_preprocesador() -> ColumnTransformer:
    """
    Devuelve un ColumnTransformer con tres transformaciones paralelas:
      - numeric_transformer: imputa mediana + estandariza
      - nominal_transformer: imputa moda + OHE
      - ordinal_transformer: imputa moda + OrdinalEncoder con orden fijo

    El preprocesador SOLO se ajusta (fit) con datos de entrenamiento.
    """

    # Pipeline numérico
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # Pipeline nominal
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',   # ignora categorías no vistas en test
            sparse_output=False,       # retorna array denso (más fácil de inspeccionar)
        )),
    ])

    # Pipeline ordinal
    # IMPORTANTE: el orden de ORDINAL_CATEGORIES debe coincidir con ORDINAL_COLS
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])

    # ColumnTransformer: aplica cada pipeline a su grupo de columnas
    # remainder='drop' → cualquier columna no listada se descarta automáticamente
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('nom', nominal_transformer, NOMINAL_COLS),
            ('ord', ordinal_transformer, ORDINAL_COLS),
            # Las binarias ya están como 0/1 → passthrough (no transformar)
            ('bin', 'passthrough', BINARY_COLS),
        ],
        remainder='drop',
        verbose_feature_names_out=False,  # nombres limpios (sin prefijo num__, nom__)
    )

    return preprocesador


# ── STEP 5: SPLIT Y CONSTRUCCIÓN FINAL ───────────────────────────────────────

def preparar_split(df: pd.DataFrame,
                   target: str,
                   test_size: float = 0.2,
                   random_state: int = 42):
    """
    Ejecuta el flujo completo:
      1. Feature engineering + codificación binaria
      2. train_test_split (80/20)
      3. Devuelve X_train, X_test, y_train, y_test

    El ColumnTransformer se ajusta FUERA de esta función (en el Pipeline del modelo)
    para garantizar que solo aprende de X_train.
    """
    X, y = preparar_dataframe(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=(y if target == TARGET_CLF else None),
        # stratify solo aplica a clasificación (para preservar la proporción 85/15)
    )

    print(f"Split completado para target='{target}':")
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    if target == TARGET_CLF:
        print(f"  Proporción riesgo_alto en train: "
              f"{y_train.mean()*100:.1f}%  |  test: {y_test.mean()*100:.1f}%")
    else:
        print(f"  Costo mediano en train: ${y_train.median():,.0f}"
              f"  |  test: ${y_test.median():,.0f}")

    return X_train, X_test, y_train, y_test


# ── UTILIDAD: nombres de features post-transformación ────────────────────────

def get_feature_names(preprocesador: ColumnTransformer) -> list[str]:
    """
    Retorna los nombres de todas las features después del ColumnTransformer.
    Útil para interpretar coeficientes y feature importances.
    """
    names = []

    # Numéricas: mismo nombre
    names += NUMERIC_COLS

    # Nominales: nombre_categoria (generado por OHE)
    ohe = preprocesador.named_transformers_['nom']['onehot']
    for col, cats in zip(NOMINAL_COLS, ohe.categories_):
        names += [f"{col}_{cat}" for cat in cats]

    # Ordinales: mismo nombre (un valor por columna)
    names += ORDINAL_COLS

    # Binarias: mismo nombre (passthrough)
    names += BINARY_COLS

    return names


# ── USO TÍPICO (ejemplo en notebook o app.py) ────────────────────────────────

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    import joblib

    df = pd.read_csv('data/seguro_auto_actuarial.csv')

    # ── Modelo de REGRESIÓN ──────────────────────────────────
    X_train, X_test, y_train, y_test = preparar_split(df, TARGET_REG)

    pipeline_reg = Pipeline(steps=[
        ('preprocessor', construir_preprocesador()),
        ('model', RandomForestRegressor(n_estimators=150, random_state=42,
                                        n_jobs=-1)),
    ])

    pipeline_reg.fit(X_train, y_train)
    joblib.dump(pipeline_reg, 'models/modelo_regresion.joblib')
    print("✅ Modelo regresión guardado.")

    # ── Modelo de CLASIFICACIÓN ──────────────────────────────
    X_train_c, X_test_c, y_train_c, y_test_c = preparar_split(df, TARGET_CLF)

    pipeline_clf = Pipeline(steps=[
        ('preprocessor', construir_preprocesador()),
        ('model', RandomForestClassifier(n_estimators=150, random_state=42,
                                          class_weight='balanced', n_jobs=-1)),
    ])

    pipeline_clf.fit(X_train_c, y_train_c)
    joblib.dump(pipeline_clf, 'models/modelo_clasificacion.joblib')
    print("✅ Modelo clasificación guardado.")

    # ── Predicción sobre datos nuevos (simulador Streamlit) ─
    # Solo necesitas pasar el DataFrame raw — el pipeline maneja todo
    nueva_poliza = pd.DataFrame([{
        'edad_conductor': 28,
        'sexo': 'Masculino',
        'estado_civil': 'Soltero',
        'nivel_estudios': 'Licenciatura',
        'ocupacion': 'Empleado',
        'zona_residencia': 'Urbana',
        'region': 'Centro',
        'antiguedad_cliente_anios': 2.5,
        'ingreso_mensual_mxn': 18000,
        'score_crediticio': 700,
        'prima_mensual_mxn': 550,
        'suma_asegurada_mxn': 280000,
        'deducible_pct': 10,
        'historial_siniestros_3_anios': 0,
        'km_anuales': 14000,
        'edad_vehiculo_anios': 3.0,
        'tipo_vehiculo': 'Sedan',
        'uso_vehiculo': 'Particular',
        'segmento_marca': 'Medio',
        'metodo_pago': 'Mensual',
        'canal_venta': 'Online',
        'tiene_gps': 'Si',
        'asistencia_vial': 'Si',
        'mantenimiento_al_dia': 'Si',
        'dias_hasta_renovacion': 120,
        'puntaje_riesgo_zona': 45.0,
        'numero_siniestros_12m': 0,
        # poliza_id, clase_costo, targets → no se necesitan para predict
    }])

    # El pipeline aplica feature_engineering + codifica internamente
    nueva_poliza = feature_engineering(nueva_poliza)
    nueva_poliza = codificar_binarias(nueva_poliza)

    costo_pred = pipeline_reg.predict(nueva_poliza)[0]
    prob_riesgo = pipeline_clf.predict_proba(nueva_poliza)[0][1]

    print(f"\n📋 Predicción para la nueva póliza:")
    print(f"   Costo esperado anual: ${costo_pred:,.0f} MXN")
    print(f"   Probabilidad riesgo alto: {prob_riesgo*100:.1f}%")
    print(f"   Clasificación: {'⚠ RIESGO ALTO' if prob_riesgo > 0.5 else '✅ Riesgo normal'}")
