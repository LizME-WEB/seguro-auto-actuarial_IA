# ============================================================
# app.py — Sistema Actuarial de Predicción de Riesgo
# Proyecto Integrador · Actuaría y Ciencia de Datos
# Ejecutar con: streamlit run app.py
# ============================================================

#importamos lo necesario y muy importante

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)
import joblib
import os

# Configuración de página (comenzamos lo nice)
st.set_page_config(
    page_title="Sistema Actuarial · Seguros de Auto",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constantes del preprocesamiento 
NUMERIC_COLS = ['edad_conductor','edad2','antiguedad_cliente_anios',
                'ingreso_mensual_mxn','log_ingreso','score_crediticio',
                'prima_mensual_mxn','suma_asegurada_mxn','deducible_pct',
                'historial_siniestros_3_anios','km_anuales','km_por_vehiculo',
                'edad_vehiculo_anios','dias_hasta_renovacion','puntaje_riesgo_zona',
                'numero_siniestros_12m','ratio_prima_suma']
NOMINAL_COLS = ['sexo','estado_civil','ocupacion','zona_residencia','region',
                'tipo_vehiculo','uso_vehiculo','metodo_pago','canal_venta']
ORDINAL_COLS = ['nivel_estudios','segmento_marca']
ORDINAL_CATS = [['Secundaria','Preparatoria','Licenciatura','Posgrado'],
                ['Economico','Medio','Premium']]
BINARY_COLS  = ['tiene_gps','asistencia_vial','mantenimiento_al_dia']
TARGET_REG   = 'costo_esperado_anual_mxn'
TARGET_CLF   = 'riesgo_alto'


# ==============================
# UTILIDADES QUE SE COMPARTIRÁN
# ==============================

@st.cache_data
def cargar_datos(path='data/seguro_auto_actuarial.csv'):#--------------cargamos la base de dtos
    df = pd.read_csv(path)
    ##---------log
    df['log_ingreso']          = np.log1p(df['ingreso_mensual_mxn'])
    ##---------edad al cuadrado
    df['edad2']                = df['edad_conductor'] ** 2
    ##---------kilometraje x vehículo
    df['km_por_vehiculo']      = df['km_anuales'] / (df['edad_vehiculo_anios'] + 1)
    df['ratio_prima_suma']     = df['prima_mensual_mxn'] / df['suma_asegurada_mxn'] * 1000
    df['tiene_gps']            = df['tiene_gps'].map({'Si':1,'No':0}).astype(float)
    df['asistencia_vial']      = df['asistencia_vial'].map({'Si':1,'No':0}).astype(float)
    df['mantenimiento_al_dia'] = df['mantenimiento_al_dia'].map(
        {'Si':1,'No':0}).fillna(-1).astype(float)
    return df


@st.cache_resource
def cargar_modelos():
    reg, clf = None, None
    try:
        reg = joblib.load('models/modelo_regresion.joblib')
    except FileNotFoundError:
        pass
    try:
        clf = joblib.load('models/modelo_clasificacion.joblib')
    except FileNotFoundError:
        pass
    return reg, clf


def build_preprocessor():
    ##-------------pipeline numérico
    num  = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler())])
    ##-------------pipeline nominal
    nom  = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore',
                                           sparse_output=False))])
    ##-------------pipeline ordinal
    ord_ = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('enc', OrdinalEncoder(categories=ORDINAL_CATS,
                                            handle_unknown='use_encoded_value',
                                            unknown_value=-1))])
    ##-------------todo en paralelo
    return ColumnTransformer([
        ('num', num,  NUMERIC_COLS),
        ('nom', nom,  NOMINAL_COLS),
        ('ord', ord_, ORDINAL_COLS),
        ('bin', 'passthrough', BINARY_COLS),
    ], remainder='drop')


def fig_to_st(fig):
    """Muestra figura matplotlib en Streamlit y la cierra."""
    st.pyplot(fig)
    plt.close(fig)


# ========
# INICIO:
# ========

def seccion_inicio(df):
    st.title(" Sistema Actuarial de Predicción de Riesgo")
    st.markdown(
        "**Proyecto Integrador · Actuaría y Ciencia de Datos** | "
        "Profe:Dr. José Alberto Guzmán Torres |" 
        "Alumnas:Lizzet Mendoza, Sofía L. Flores y A. Berenice García"
    )
    st.divider()

    # Métricas
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pólizas", f"{len(df):,}")
    c2.metric("Variables", "31")
    c3.metric("Riesgo Alto", f"{df[TARGET_CLF].mean()*100:.1f}%")
    c4.metric("Costo mediano", f"${df[TARGET_REG].median():,.0f}")
    c5.metric("Faltantes en datos", "6 columnas")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(" ¿Qué predice este sistema?")
        st.info(
            "**Regresión:** Estimar el `costo_esperado_anual_mxn` — cuánto costará "
            "siniestrar una póliza durante un año.\n\n"
            "**Clasificación:** Predecir si una póliza pertenece al grupo de "
            "`riesgo_alto` (clase desbalanceada: 15%)."
        )
    with col_b:
        st.subheader(" Contexto actuarial")
        st.markdown(
            "Una aseguradora necesita estimar el costo esperado de siniestros para "
            "**tarifar correctamente** cada póliza. Si cobra de menos, pierde dinero; "
            "si cobra de más, pierde al cliente. El clasificador permite **segmentar** "
            "la cartera y aplicar condiciones diferenciadas al grupo de mayor riesgo."
        )

    st.subheader(" Vista previa de los datos")
    st.dataframe(df.drop(columns=['poliza_id']).head(10),
                 use_container_width=True, height=280)

    with st.expander(" Diccionario de variables principales"):
        diccionario = {
            'Variable': ['edad_conductor','ingreso_mensual_mxn','suma_asegurada_mxn',
                          'numero_siniestros_12m','historial_siniestros_3_anios',
                          'score_crediticio','prima_mensual_mxn','tipo_vehiculo',
                          'uso_vehiculo','riesgo_alto','costo_esperado_anual_mxn'],
            'Tipo': ['Numérica','Numérica','Numérica','Numérica','Numérica',
                     'Numérica','Numérica','Nominal','Nominal','Target clf','Target reg'],
            'Descripción': [
                'Edad del conductor principal (18–75 años)',
                'Ingreso mensual estimado (con faltantes y outliers)',
                'Valor asegurado del vehículo (109k–1.6M MXN)',
                'Siniestros ocurridos en los últimos 12 meses',
                'Número de siniestros previos en 3 años',
                'Puntaje crediticio (401–850)',
                'Prima mensual de la póliza',
                'Sedán, SUV, Compacto, Pickup, Deportivo',
                'Particular, Trabajo, Taxi/Plataforma',
                '1 = riesgo alto (15% de la cartera)',
                'Costo anual esperado de siniestros (MXN)',
            ]
        }
        st.dataframe(pd.DataFrame(diccionario), use_container_width=True,
                     hide_index=True)


# =========================
# EXPLORACIÓN DE LOS DATOS:
# =========================

def seccion_eda(df):
    st.title(" Exploración de Datos")
    st.markdown("Análisis descriptivo de la cartera de pólizas antes del modelado.")

    # - Faltantes:
    st.subheader("Valores faltantes")
    faltantes = df.isnull().sum() #------------cuenta los nulos por columna
    faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
    pct_falt  = (faltantes / len(df) * 100).round(1)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(
            pd.DataFrame({'Columna': faltantes.index,
                          'N faltantes': faltantes.values,
                          '% faltantes': pct_falt.values}),
            hide_index=True, use_container_width=True
        )
    with col2:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(faltantes.index, pct_falt.values,
                color='#E24B4A', edgecolor='none')
        ax.set_xlabel('% de valores faltantes')
        ax.set_title('Columnas con valores faltantes')
        for i, v in enumerate(pct_falt.values):
            ax.text(v + 0.1, i, f'{v}%', va='center', fontsize=9)
        plt.tight_layout()
        fig_to_st(fig)

    # - Balance de las clases:
    st.subheader("Balance de la clase objetivo (riesgo_alto)")
    col3, col4 = st.columns(2)
    with col3:
        vc = df[TARGET_CLF].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(['Normal (0)', 'Riesgo Alto (1)'], vc.values,
               color=['#9FE1CB', '#E24B4A'], edgecolor='none')
        ax.set_title('Balance de clases: riesgo_alto')
        for i, v in enumerate(vc.values):
            ax.text(i, v + 10, f'{v}\n({v/len(df)*100:.1f}%)',
                    ha='center', fontsize=10)
        ax.set_ylim(0, 1450)
        plt.tight_layout()
        fig_to_st(fig)
        st.warning("⚠ Clase desbalanceada (85/15). Usar recall y F1, no solo accuracy.")
    with col4:
        vc2 = df['clase_costo'].value_counts()
        colores = {'Bajo':'#9FE1CB','Medio':'#EF9F27','Alto':'#E24B4A'}
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(vc2.index, vc2.values,
                      color=[colores.get(k,'#378ADD') for k in vc2.index],
                      edgecolor='none')
        ax.set_title('Distribución: clase_costo')
        for i, (lbl, v) in enumerate(vc2.items()):
            ax.text(i, v + 10, f'{v}', ha='center', fontsize=10)
        plt.tight_layout()
        fig_to_st(fig)

    # ─ Distribuciones numé.:
    st.subheader("Distribuciones de variables numéricas")
    cols_plot = ['costo_esperado_anual_mxn','ingreso_mensual_mxn',
                 'prima_mensual_mxn','suma_asegurada_mxn',
                 'km_anuales','score_crediticio']
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, col in zip(axes.flatten(), cols_plot):
        ax.hist(df[col].dropna(), bins=40, color='#378ADD',
                edgecolor='none', alpha=0.8)
        skew = df[col].skew()#------------------------CALCULA EL SESGO DE CADA VARIABLE
        color = '#E24B4A' if abs(skew) > 2 else ('#EF9F27' if abs(skew) > 0.5 else '#3B6D11')
        ax.set_title(f'{col}\nskew = {skew:.2f}', fontsize=9, color=color)
        ax.tick_params(labelsize=8)
    plt.suptitle('Distribuciones — rojo: sesgo extremo, verde: simétrica', fontsize=10)
    plt.tight_layout()
    fig_to_st(fig)

    # - Correlaciones:
    st.subheader("Matriz de correlaciones")
    num_cols_corr = ['edad_conductor','ingreso_mensual_mxn','score_crediticio',
                     'prima_mensual_mxn','suma_asegurada_mxn','km_anuales',
                     'historial_siniestros_3_anios','numero_siniestros_12m',
                     TARGET_REG, TARGET_CLF]
    corr = df[num_cols_corr].corr()#-------------------CALCULA LA MATRIZ DE CORRELACIÓN
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.4, annot_kws={'fontsize': 9})
    ax.set_title('Correlaciones entre variables numéricas', fontsize=11)
    plt.tight_layout()
    fig_to_st(fig)

    # - Categóricas vs targets:
    st.subheader("Variables categóricas vs targets")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ['tipo_vehiculo','uso_vehiculo','zona_residencia']):
        orden = (df.groupby(col)[TARGET_REG]
                   .median().sort_values(ascending=False))
        ax.bar(orden.index, orden.values, color='#378ADD', edgecolor='none')
        ax.set_title(f'{col}\nvs costo mediano', fontsize=10)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    plt.suptitle('Variables categóricas vs Costo esperado mediano', fontsize=11)
    plt.tight_layout()
    fig_to_st(fig)


# =================
# PREPROCESAMIENTO:
# =================

def seccion_preprocesamiento(df):
    st.title("️ Preprocesamiento")
    ##-------
    st.markdown("Transformaciones aplicadas antes del entrenamiento.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clasificación de variables")
        tabla = pd.DataFrame({
            'Tipo': ['Numéricas (17)', 'Nominales (9)', 'Ordinales (2)', 'Binarias (3)', 'Excluidas'],
            'Transformación': ['Mediana + StandardScaler', 'Moda + OneHotEncoder',
                               'Moda + OrdinalEncoder', 'Mapeo Si=1/No=0', 'Sin uso'],
            'Ejemplos': ['suma_asegurada, km_anuales', 'tipo_vehiculo, uso_vehiculo',
                         'nivel_estudios, segmento_marca', 'tiene_gps, asistencia_vial',
                         'poliza_id, clase_costo'],
        })
        st.dataframe(tabla, hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Features engineeradas (nuevas)")
        tabla_fe = pd.DataFrame({
            'Feature': ['log_ingreso', 'edad2', 'km_por_vehiculo', 'ratio_prima_suma'],
            'Fórmula': ['log(ingreso + 1)', 'edad²', 'km / (edad_veh + 1)',
                        'prima / suma × 1000'],
            'Justificación': ['Corrige sesgo 5.67→0.31', 'Captura curva en U',
                              'Intensidad de uso', 'Tarificación relativa'],
        })
        st.dataframe(tabla_fe, hide_index=True, use_container_width=True)

    st.subheader("Codificación ordinal con orden explícito")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**nivel_estudios** → Secundaria(0) → Preparatoria(1) → Licenciatura(2) → Posgrado(3)")
    with col4:
        st.markdown("**segmento_marca** → Económico(0) → Medio(1) → Premium(2)")

    st.subheader("Arquitectura del Pipeline")
    st.code("""
preprocessor = ColumnTransformer([
    ('num', Pipeline([SimpleImputer(median), StandardScaler()]),    numeric_cols),
    ('nom', Pipeline([SimpleImputer(moda),   OneHotEncoder()]),     nominal_cols),
    ('ord', Pipeline([SimpleImputer(moda),   OrdinalEncoder()]),    ordinal_cols),
    ('bin', 'passthrough',                                          binary_cols),
])

pipeline_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model',        Lasso(alpha=50))
])

pipeline_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model',        GradientBoostingClassifier(n_estimators=150))
])
    """, language='python')

    X = df.drop(columns=['poliza_id','clase_costo', TARGET_REG, TARGET_CLF])
    X_tr, X_te, _, _ = train_test_split(X, df[TARGET_REG],
                                         test_size=0.2, random_state=42)
    pre = build_preprocessor()
    X_tr_t = pre.fit_transform(X_tr)

    c1, c2, c3 = st.columns(3)
    c1.metric("Features originales", X.shape[1])
    c2.metric("Features post-OHE", X_tr_t.shape[1])
    c3.metric("Filas entrenamiento", X_tr.shape[0])

    st.info(
        " El Pipeline garantiza que el escalador y el imputador solo aprenden "
        "del training set. Al llamar `pipeline.predict()` en el simulador, se aplica "
        "exactamente la misma transformación."
    )


# ═========
# MODELADO:(
# =========

def seccion_modelado(df, pipe_reg, pipe_clf):
    st.title(" Modelado y Evaluación")
    #------se preparan los datos para la regresión
    
####---------Split Estratificado proporción 85/15 en train y test sets
    # Preparamos los datitos:
    X_reg = df.drop(columns=['poliza_id','clase_costo', TARGET_CLF, TARGET_REG])
    y_reg = df[TARGET_REG]
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg,
                                               test_size=0.2, random_state=42)
    X_clf = df.drop(columns=['poliza_id','clase_costo', TARGET_REG, TARGET_CLF])
    y_clf = df[TARGET_CLF]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_clf, y_clf,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y_clf)

    # ─ Botón para entrenar:
    if pipe_reg is None or pipe_clf is None:
        st.warning("⚠ No se encontraron modelos guardados. Entrena los modelos aquí.")
        if st.button(" Entrenar todos los modelos", type='primary'):
            with st.status("Entrenando modelos...", expanded=True) as status:
                os.makedirs('models', exist_ok=True)

                st.write("Entrenando regresión lineal...")
                pipe_reg = Pipeline([('pre', build_preprocessor()),
                                     ('mod', Lasso(alpha=50, max_iter=5000))])
                pipe_reg.fit(X_tr, y_tr)
                joblib.dump(pipe_reg, 'models/modelo_regresion.joblib')

                st.write("Entrenando GradientBoosting clasificación...")
                pipe_clf = Pipeline([('pre', build_preprocessor()),
                                     ('mod', GradientBoostingClassifier(
                                         n_estimators=150, random_state=42))])
                pipe_clf.fit(Xc_tr, yc_tr)
                joblib.dump(pipe_clf, 'models/modelo_clasificacion.joblib')

                status.update(label=" Modelos entrenados y guardados", state="complete")
            st.cache_resource.clear()
            st.rerun()
        return

    # - Tab de regresión vs clasificación:
    tab_reg, tab_clf = st.tabs([" Regresión", " Clasificación"])

    with tab_reg:
        st.subheader("Comparativa de modelos de regresión")

        modelos_reg = {
            'LinearRegression': LinearRegression(),
            'Ridge (α=1)':      Ridge(alpha=1.0),
            'Lasso (α=50)':     Lasso(alpha=50, max_iter=5000),#pipeline que se evalúa en el ciclo
            'DTree (d=6)':      DecisionTreeRegressor(max_depth=6, random_state=42),
            'RandomForest':     RandomForestRegressor(n_estimators=100,
                                                       random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100,
                                                           random_state=42),
        }

        rows = []
        with st.spinner("Evaluando modelos de regresión..."):
            for nombre, modelo in modelos_reg.items():
                p = Pipeline([('pre', build_preprocessor()), ('mod', modelo)])
                p.fit(X_tr, y_tr)
                yp = p.predict(X_te)

    ###--------las 3 métricas
                rows.append({
                    'Modelo': nombre,
                    'MAE ($)': f"${mean_absolute_error(y_te, yp):,.0f}",
                    'RMSE ($)': f"${np.sqrt(mean_squared_error(y_te, yp)):,.0f}",
                    'R²': round(r2_score(y_te, yp), 4),
                })

        df_reg_res = pd.DataFrame(rows)
        st.dataframe(df_reg_res.style.highlight_max(subset=['R²'], color='#C0DD97')
                                      .highlight_min(subset=['R²'], color='#FCEBEB'),
                     hide_index=True, use_container_width=True)

        

        # Feature importance del modelo guardado (RF o Lasso)
        
        st.subheader("Interpretación — coeficientes del modelo Lasso (α=50)")
     ###------pipeline, construcción lasso 
        lasso_p = Pipeline([('pre', build_preprocessor()),
                             ('mod', Lasso(alpha=50, max_iter=5000))])
        lasso_p.fit(X_tr, y_tr)
        coefs  = lasso_p.named_steps['mod'].coef_
        pre_l  = lasso_p.named_steps['pre']
        ohe_l  = pre_l.named_transformers_['nom']['ohe'].categories_
        nom_l  = [f"{c}_{v}" for c, cats in zip(NOMINAL_COLS, ohe_l) for v in cats]
        f_names = NUMERIC_COLS + nom_l + ORDINAL_COLS + BINARY_COLS

     ###-------selección automática, filtra
        activas = [(f_names[i], coefs[i]) for i in range(len(coefs)) if coefs[i] != 0]
        activas_s = sorted(activas, key=lambda x: abs(x[1]), reverse=True)[:12]

        fig, ax = plt.subplots(figsize=(9, 5))
        nombres_c = [a[0] for a in activas_s]
        valores_c = [a[1] for a in activas_s]
        colores_c = ['#185FA5' if v > 0 else '#E24B4A' for v in valores_c]
        ax.barh(nombres_c[::-1], valores_c[::-1], color=colores_c[::-1],
                edgecolor='none')
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('Coeficiente (escala estandarizada)')
        ax.set_title(f'Lasso(α=50) — {len(activas)} de {len(coefs)} features activas')
        plt.tight_layout()
        fig_to_st(fig)
        
     ###--------pié de gráfica, variables irelevantes
        st.caption(
            "Azul = efecto positivo sobre el costo · "
            "Rojo = efecto negativo · "
            "Features con coeficiente = 0 fueron eliminadas automáticamente por Lasso."
        )

    with tab_clf:
        st.subheader("Comparativa de modelos de clasificación")
        st.info(
            "Con clase desbalanceada (15%), el modelo naive que siempre predice "
            "'Normal' tendría 85% de accuracy. **Recall y F1 son las métricas clave.**"
        )

        modelos_clf = {
   ####-----------RANDOM FOREST BALANCEADO
            'DTree (d=3) balanced': DecisionTreeClassifier(max_depth=3,
                                     class_weight='balanced', random_state=42),
            'DTree (d=6) balanced': DecisionTreeClassifier(max_depth=6,
                                     class_weight='balanced', random_state=42),
            'RF balanced':          RandomForestClassifier(n_estimators=100,
                                     class_weight='balanced', random_state=42,
                                     n_jobs=-1),
            'RF sin balance':       RandomForestClassifier(n_estimators=100,
                                     random_state=42, n_jobs=-1),
   ####------BOOSTING, El modelo secuencial que gana el mejor F1
            'GradientBoosting':     GradientBoostingClassifier(n_estimators=100,
                                     random_state=42),
        }

        rows_c = []
        with st.spinner("Evaluando modelos de clasificación..."):
            for nombre, modelo in modelos_clf.items():
                p = Pipeline([('pre', build_preprocessor()), ('mod', modelo)])
                p.fit(Xc_tr, yc_tr)
                yp = p.predict(Xc_te)
                
    ####--------MÉTRICAS DE EVALUACIÓN
                rows_c.append({
                    'Modelo': nombre,
                    'Accuracy': round(accuracy_score(yc_te, yp), 3),
                    'Precision': round(precision_score(yc_te, yp, zero_division=0), 3),
                    'Recall': round(recall_score(yc_te, yp, zero_division=0), 3),
                    'F1': round(f1_score(yc_te, yp, zero_division=0), 3),
                })

        df_clf_res = pd.DataFrame(rows_c)
        st.dataframe(df_clf_res.style.highlight_max(subset=['F1','Recall'],
                                                      color='#C0DD97'),
                     hide_index=True, use_container_width=True)

        # Matriz de confusión del modelo cargado
        
        st.subheader("Matriz de confusión — modelo activo")
        yp_final = pipe_clf.predict(Xc_te)
    ####----------matriz de confusion, desglosa aciertos, falsos positivos y falsos negativos
        cm = confusion_matrix(yc_te, yp_final)

        col_cm1, col_cm2 = st.columns([1, 1])
        with col_cm1:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                           display_labels=['Normal', 'Riesgo Alto'])
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title('Matriz de confusión')
            plt.tight_layout()
            fig_to_st(fig)
        with col_cm2:
            fn = cm[1, 0]
            fp = cm[0, 1]
            tp = cm[1, 1]
            tn = cm[0, 0]
            st.metric("Verdaderos Negativos (TN)", tn)
            st.metric("Falsos Positivos (FP)", fp,
                      help="Pólizas normales clasificadas como riesgo alto")
            st.metric("Falsos Negativos (FN)", fn,
                      delta=f"-{fn} sin detectar",
                      delta_color="inverse",
                      help="⚠ Pólizas peligrosas no detectadas — costo real para la aseguradora")
            st.metric("Verdaderos Positivos (TP)", tp)


# =====
# PCA:
# =====

def seccion_pca(df):
    st.title(" Reducción de Dimensionalidad — PCA")

    X = df.drop(columns=['poliza_id','clase_costo', TARGET_REG, TARGET_CLF])
    pre = build_preprocessor()

    with st.spinner("Aplicando preprocesamiento y PCA..."):
 #*------transformación previa
        X_t = pre.fit_transform(X)

    # ─ Varianza explicada:
    
    st.subheader("Varianza explicada por componente")
    pca_full = PCA(random_state=42)
    pca_full.fit(X_t)
    var_ratio = pca_full.explained_variance_ratio_
 #*---------CUMSUM Calcula la varianza acumulada para saber dónde corta el 95%
    var_acum  = np.cumsum(var_ratio)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(1, 21), var_ratio[:20] * 100,
               color='#378ADD', edgecolor='none', alpha=0.85)
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Varianza explicada (%)')
        ax.set_title('Varianza individual (Scree plot)')
        plt.tight_layout()
        fig_to_st(fig)
    with col2:#---varianza acum
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(var_acum)+1), var_acum * 100,
                color='#185FA5', linewidth=2)
        ax.fill_between(range(1, len(var_acum)+1),
                         var_acum * 100, alpha=0.15, color='#185FA5')
 #*-----------MARCADOR DEL 95%! Graf
        for thresh, color, label in [
            (0.80,'#3B6D11','80%'), (0.90,'#EF9F27','90%'), (0.95,'#E24B4A','95%')]:
            n = int(np.argmax(var_acum >= thresh)) + 1
            ax.axhline(thresh*100, color=color, linestyle='--',
                       linewidth=1, label=f'{label} → {n} componentes')
        ax.set_xlabel('Número de componentes')
        ax.set_ylabel('Varianza acumulada (%)')
        ax.set_title('Varianza acumulada')
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig_to_st(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Varianza PC1", f"{var_ratio[0]*100:.2f}%")
    c2.metric("Varianza PC2", f"{var_ratio[1]*100:.2f}%")
    c3.metric("Total 2D", f"{(var_ratio[0]+var_ratio[1])*100:.2f}%")

    # ─ Visualización 2D: modo pro jij
    st.subheader("Visualización PCA 2D")
    color_by = st.radio("Colorear por:", ['riesgo_alto', 'clase_costo'],
                         horizontal=True)
 #*----------PCA 2, de 2 componentes Reduce la dimensionalidad para la visualización en el plano
    pca2   = PCA(n_components=2, random_state=42)
    X_pca  = pca2.fit_transform(X_t)
    var1   = pca2.explained_variance_ratio_[0] * 100
    var2   = pca2.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    if color_by == 'riesgo_alto':
        paleta = {0: '#9FE1CB', 1: '#E24B4A'}
        labels = {0: f'Normal (n={int((df[TARGET_CLF]==0).sum())})',
                  1: f'Riesgo Alto (n={int((df[TARGET_CLF]==1).sum())})'}
        for clase in [0, 1]:
            mask = df[TARGET_CLF].values == clase
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=paleta[clase], alpha=0.4 if clase == 0 else 0.7,
                       s=8 if clase == 0 else 12,
                       label=labels[clase], edgecolors='none')
            cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
            ax.scatter(cx, cy, c=paleta[clase], s=120, marker='D',
                       edgecolors='white', linewidths=1.5, zorder=5)
    else:
        paleta = {'Bajo':'#9FE1CB','Medio':'#EF9F27','Alto':'#E24B4A'}
        for clase, color in paleta.items():
            mask = df['clase_costo'].values == clase
            n = mask.sum()
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, alpha=0.4, s=8, edgecolors='none',
                       label=f'{clase} (n={n})')

    ax.set_xlabel(f'PC1 ({var1:.1f}% varianza)')
    ax.set_ylabel(f'PC2 ({var2:.1f}% varianza)')
    ax.set_title(f'PCA 2D — coloreado por {color_by}')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    fig_to_st(fig)

 #*---------------INTERPRETACIÓN DE LOS LOADINGS
    
    st.caption(
        f"PC1 (eje edad): loading edad_conductor = +0.64. "
        f"PC2 (eje uso/ingresos): loading km_por_vehiculo = +0.41, "
        f"log_ingreso = −0.38. "
        f"Total varianza capturada en 2D: {var1+var2:.1f}%."
    )
    st.info(
        " Los centroides de 'Riesgo Alto' y 'Alto costo' se desplazan hacia el "
        "cuadrante superior-izquierdo (PC2 positivo), indicando mayor intensidad "
        "de uso del vehículo. El solapamiento es esperado — 2 componentes solo "
        "capturan el 16% de la varianza."
    )


# ====================
# SIMULADOR ACTUARIAL
# ====================

def seccion_simulador(pipe_reg, pipe_clf):
    st.title(" Simulador Actuarial")
    st.markdown(
        "Ingresa los datos de una póliza nueva para obtener el **costo esperado anual** "
        "y la **probabilidad de ser riesgo alto**."
    )

    if pipe_reg is None or pipe_clf is None:
        st.error(" Modelos no encontrados. Ve a la sección **Modelado** y entrena primero.")
        return

    # ─ Formulario organizado en 3 columns:
    with st.form("simulador_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("** Datos del conductor**")
            edad          = st.slider("Edad del conductor", 18, 75, 35)
            sexo          = st.selectbox("Sexo", ['Masculino','Femenino','No especificado'])
            estado_civil  = st.selectbox("Estado civil", ['Soltero','Casado','Divorciado','Viudo'])
            nivel_est     = st.selectbox("Nivel de estudios",
                                          ['Secundaria','Preparatoria','Licenciatura','Posgrado'],
                                          index=2)
            ocupacion     = st.selectbox("Ocupación",
                                          ['Empleado','Empresario','Estudiante',
                                           'Independiente','Otro','Profesor'])
            ingreso       = st.number_input("Ingreso mensual (MXN)", 6000, 350000,
                                             20000, step=1000)
            score         = st.slider("Score crediticio", 401, 850, 650)
            antiguedad    = st.slider("Antigüedad como cliente (años)", 0, 20, 3)
            

        with col2:
            st.markdown("** Datos del vehículo**")
            tipo_veh      = st.selectbox("Tipo de vehículo",
                                          ['Sedan','SUV','Compacto','Pickup','Deportivo'])
            uso_veh       = st.selectbox("Uso del vehículo",
                                          ['Particular','Trabajo','Taxi/Plataforma'])
            segmento      = st.selectbox("Segmento de marca",
                                          ['Economico','Medio','Premium'])
            suma_aseg     = st.number_input("Suma asegurada (MXN)",
                                             110000, 1600000, 280000, step=10000)
            edad_veh      = st.slider("Antigüedad del vehículo (años)", 0, 20, 4)
            km            = st.number_input("Km anuales recorridos",
                                             2500, 40000, 15000, step=500)
            tiene_gps_v   = st.radio("¿Tiene GPS?", ['Si','No'], horizontal=True)
            asist_vial    = st.radio("¿Asistencia vial?", ['Si','No'], horizontal=True)
            

        with col3:
            st.markdown("** Datos de la póliza**")
            zona          = st.selectbox("Zona de residencia",
                                          ['Urbana','Suburbana','Rural'])
            region        = st.selectbox("Región",
                                          ['Centro','Norte','Sur','Bajio','Occidente'])
            prima         = st.number_input("Prima mensual (MXN)",
                                             350, 3500, 500, step=50)
            deducible     = st.selectbox("Deducible (%)", [3, 5, 10, 15])
            metodo_pago   = st.selectbox("Método de pago",
                                          ['Mensual','Trimestral','Anual'])
            canal_venta   = st.selectbox("Canal de venta",
                                          ['Agente','Online','Banco','Broker'])
            mantenimiento = st.radio("¿Mantenimiento al día?",
                                      ['Si','No'], horizontal=True)
            puntaje_zona  = st.slider("Puntaje riesgo de zona", 22, 90, 55)
            hist_siniest  = st.slider("Siniestros previos (3 años)", 0, 5, 0)
            siniest_12m   = st.slider("Siniestros últimos 12 meses", 0, 3, 0)
            dias_renov    = st.slider("Días hasta renovación", 0, 365, 180)

        submitted = st.form_submit_button(" Calcular predicción", type='primary',
                                           use_container_width=True)

    # ─ Predicción:
    if submitted:
 #*------------DATAFRAME Agrupa todos los inputs del formulario en una sola fila
        poliza = pd.DataFrame([{
            'edad_conductor':            edad,
            'sexo':                      sexo,
            'estado_civil':              estado_civil,
            'nivel_estudios':            nivel_est,
            'ocupacion':                 ocupacion,
            'zona_residencia':           zona,
            'region':                    region,
            'antiguedad_cliente_anios':  antiguedad,
            'ingreso_mensual_mxn':       ingreso,
            'score_crediticio':          score,
            'prima_mensual_mxn':         prima,
            'suma_asegurada_mxn':        suma_aseg,
            'deducible_pct':             deducible,
            'historial_siniestros_3_anios': hist_siniest,
            'km_anuales':                km,
            'edad_vehiculo_anios':       edad_veh,
            'tipo_vehiculo':             tipo_veh,
            'uso_vehiculo':              uso_veh,
            'segmento_marca':            segmento,
            'metodo_pago':               metodo_pago,
            'canal_venta':               canal_venta,
            'tiene_gps':                 1 if tiene_gps_v == 'Si' else 0,
            'asistencia_vial':           1 if asist_vial == 'Si' else 0,
            'mantenimiento_al_dia':      1 if mantenimiento == 'Si' else (0 if mantenimiento == 'No' else -1),
            'dias_hasta_renovacion':     dias_renov,
            'puntaje_riesgo_zona':       puntaje_zona,
            'numero_siniestros_12m':     siniest_12m,
        }])


        # Feature engineering igual que en entrenamiento
  #*-----------FÓRMULAS
        poliza['log_ingreso']      = np.log1p(poliza['ingreso_mensual_mxn'])
        poliza['edad2']            = poliza['edad_conductor'] ** 2
        poliza['km_por_vehiculo']  = poliza['km_anuales'] / (poliza['edad_vehiculo_anios'] + 1)
        poliza['ratio_prima_suma'] = poliza['prima_mensual_mxn'] / poliza['suma_asegurada_mxn'] * 1000

  #*------pipe_reg ejecuta automáticamente el ColumnTransformer (imputación + StandardScaler) y estima el costo en pesos
        costo_pred = pipe_reg.predict(poliza)[0]
  #*-------pipe_clf transforma las columnas categóricas (OHE) y calcula la probab CLASE 1
        prob_riesgo = pipe_clf.predict_proba(poliza)[0][1]
        clase_pred  = pipe_clf.predict(poliza)[0]

        st.divider()
        st.subheader(" Resultado de la predicción")
        r1, r2, r3 = st.columns(3)

        r1.metric(" Costo esperado anual", f"${costo_pred:,.0f} MXN",
                   help="Estimación del costo de siniestros para esta póliza")
        r2.metric(" Probabilidad riesgo alto", f"{prob_riesgo*100:.1f}%",
                   help="Probabilidad de pertenecer al grupo de mayor siniestralidad")

        if clase_pred == 1:
            r3.error("⚠ RIESGO ALTO\nSe recomienda revisar condiciones de la póliza")
        else:
            r3.success(" RIESGO NORMAL\nPóliza dentro del perfil estándar")

        # Gauge visual de probabilidad
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([''], [prob_riesgo * 100], color='#E24B4A', height=0.5)
        ax.barh([''], [100 - prob_riesgo * 100], left=[prob_riesgo * 100],
                color='#EAF3DE', height=0.5)
        ax.set_xlim(0, 100)
        ax.axvline(15, color='gray', linestyle='--', linewidth=1.2,
                   label='Umbral promedio (15%)')
        ax.set_xlabel('Probabilidad de riesgo alto (%)')
        ax.legend(fontsize=9)
        ax.set_title(f'Probabilidad estimada: {prob_riesgo*100:.1f}%')
        plt.tight_layout()
        fig_to_st(fig)

        with st.expander(" Ver datos de la póliza enviados al modelo"):
            st.dataframe(poliza.T.rename(columns={0: 'Valor'}),
                         use_container_width=True)


# ==========
# IMÁGENES:
# ==========

def seccion_imagenes():
    st.title(" Módulo de Análisis de Imágenes")
    st.markdown(
        "Carga una fotografía de un **vehículo** o **evidencia de siniestro** "
        "para explorar transformaciones con OpenCV (Capítulo 8 del libro)."
    )

    try:
        import cv2
        from PIL import Image
    except ImportError:
        st.error("Instala OpenCV: `pip install opencv-python-headless`")
        return

    archivo = st.file_uploader("Sube una imagen (jpg, jpeg, png)",
                                type=["jpg", "jpeg", "png"])
    if archivo is None:
        st.info(" Sube una foto de vehículo para comenzar el análisis.")
        return

    pil_img = Image.open(archivo).convert('RGB')
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_rgb.shape[:2]


    # Parámetros en sidebar
    st.sidebar.markdown("### ️ Parámetros")
    rw = st.sidebar.slider("Resize ancho", 64, 512, 256, 32)
    rh = st.sidebar.slider("Resize alto",  64, 512, 192, 32)
    blur_k = st.sidebar.slider("Blur kernel", 3, 31, 11, 2)
    sharp_i = st.sidebar.slider("Sharp intensidad", 1, 5, 2)
    crop_p  = st.sidebar.slider("Crop % desde arriba", 0, 40, 20, 5)
#**-----------SUAVIZADO / Blur Gaussiano, mayor tamaño de kernel, más desenfoque
    k = blur_k if blur_k % 2 == 1 else blur_k + 1
#**------------RESIZE Cambia las dimensiones usando el ancho y alto del slider
    img_resize   = cv2.resize(img_rgb, (rw, rh))
    img_blur     = cv2.GaussianBlur(img_rgb, (k, k), 0)
#**-----------REALCE DE BORDES (SHARPENING)
    sk = np.array([[0, -sharp_i, 0], [-sharp_i, 4*sharp_i+1, -sharp_i], [0, -sharp_i, 0]])
    img_sharp    = np.clip(cv2.filter2D(img_rgb, -1, sk), 0, 255).astype(np.uint8)
    y0 = int(h * crop_p / 100)
    img_crop     = img_rgb[y0:, :]
#**-----------Convierte la imagen de BGR al espacio YUV
    img_yuv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
#**-----------ecualización de histograma únicamente al canal 0
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#**-----------Regresa la imagen
    img_contrast = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

#**-----------mediana estadística de la imagen
    med          = np.median(img_gray)
#**-----------Canny calcula dinámicamente sus umbrales usando un rango
    img_edges    = cv2.Canny(img_gray, int(max(0,(1-0.33)*med)),
                             int(min(255,(1+0.33)*med)))

    st.subheader("Transformaciones aplicadas")
    c1, c2, c3 = st.columns(3)
    c1.image(img_rgb,      caption=f"Original ({w}×{h})",    use_column_width=True)
    c1.image(img_blur,     caption=f"Blur Gaussiano ({k}×{k})", use_column_width=True)
    c2.image(img_resize,   caption=f"Resize {rw}×{rh}",      use_column_width=True)
    c2.image(img_sharp,    caption=f"Sharpening (int={sharp_i})", use_column_width=True)
    c3.image(img_crop,     caption=f"Crop ({crop_p}% recortado)", use_column_width=True)
    c3.image(img_contrast, caption="Contraste (equalizeHist)", use_column_width=True)
    st.image(img_edges, caption="Detección de bordes — Canny",
             channels="GRAY", use_column_width=True)



    # Histograma de color
    st.subheader("Histograma de distribución de color (RGB)")
    fig, ax = plt.subplots(figsize=(10, 3))
    for i, (color, nombre) in enumerate([('#E24B4A','Rojo'),('#3B6D11','Verde'),('#185FA5','Azul')]):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, linewidth=1.2, alpha=0.85, label=nombre)
    ax.set_xlim([0, 256])
    ax.set_xlabel('Intensidad del píxel')
    ax.set_ylabel('Número de píxeles')
    ax.set_title('Histograma de color')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_to_st(fig)


# =============
# CONCLUSIONES
# =============

def seccion_conclusiones():
    st.title(" Conclusiones")

    st.subheader("Hallazgos")
    hallazgos = [
        ("Los modelos lineales superaron al Random Forest en regresión porque pudimos obtener (R²=0.43 vs 0.34)",
         "La relación suma_asegurada → costo es aproximadamente lineal. Más complejidad no garantiza mejor desempeño."),
        ("El desbalance de clases (85/15) hace que accuracy sea una métrica tramposita",
         "Un modelo naive que prediga siempre 'Normal' tendría 85% de accuracy pero cero utilidad 100% real."),
        ("GradientBoosting logró el mejor F1=0.747 en clasificación",
         "El entrenamiento secuencial enfoca cada árbol en los errores previos, siendo más efectivo con clases menores."),
        ("Las 4 features engineeradas tienen impacto medible",
         "edad², km_por_vehiculo, log_ingreso y ratio_prima_suma aparecen en los top 10 de importancia."),
        ("PCA 2D captura solo el 16% de varianza",
         "Para reducción real se necesitan 34 componentes (95%). El 2D es solo para visualización exploratoria y de visualización."),
    ]
    for titulo, desc in hallazgos:
        with st.expander(f" {titulo}"):
            st.write(desc)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Limitaciones")
        st.markdown("""
        - Base de datos **supuesta** — resultados no generalizables a datos reales
        - No se incorporaron factores externos (clima, estado de carreteras, hora del día, y varias cosas mas)
        - El clasificador tiene recall=0.40 en el RF balanced — aún pierde muchos riesgos altos
        - Las variables socioeconómicas (ingreso, ocupación) pueden introducir **sesgos discriminatorios** aunque podemos decir que para esto se hace un análisis más cauteloso
        """)
    with col2:
        st.subheader(" Mejoras")
        st.markdown("""
        - Ajuste de umbral de clasificación (0.5 → 0.3) para mejorar recall
        - XGBoost / LightGBM como alternativa a GradientBoosting
        - SHAP values para explicabilidad individual de predicciones
        - Validación cruzada estratificada para estimaciones de mayores datos
        - Datos reales con validación, porque qunque hay sesgos, tratandose de este tipo de riesgos, pólizas y temas de cálculo, si se consideran más datos
        - Modelo de segmentación (clustering) como paso previo a la tarificación
        """)

    st.subheader(" Consideraciones éticas (importante)")
    st.warning(
        "El uso de variables como `sexo`, `nivel_estudios` e `ingreso_mensual` en "
        "modelos de tarificación puede amplificar desigualdades socioeconómicas. "
        "En México, la CONDUSEF regula el uso de variables discriminatorias en seguros, tmbn, leimos un articulo en que pueden sancionarse algunos detalles tratandose de genero. "
        "Todo modelo en producción debe auditarse periódicamente para detectar sesgos "
        "y validarse con datos reales antes de tomar decisiones sobre clientes reales."
    )


# =====================
# NAVEGACIÓN PRINCIPAL
# =====================

def main():
    df = cargar_datos()
    pipe_reg, pipe_clf = cargar_modelos()

    # Sidebar de navegación
    with st.sidebar:
        st.markdown("##  Seguros de Auto")
        st.markdown("Sistema Actuarial ")
        st.divider()

        pagina = st.radio(
            "Navegación",
            options=[
                " Inicio",
                " Exploración de datos",
                "️ Preprocesamiento",
                " Modelado",
                " Reducción dimensional (PCA)",
                " Simulador actuarial",
                " Módulo de imágenes",
                " Conclusiones",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        estado_reg = " Cargado :)" if pipe_reg else " No encontrado:( "
        estado_clf = " Cargado :)" if pipe_clf else " No encontrado:( "
        st.caption(f"Modelo regresión: {estado_reg}")
        st.caption(f"Modelo clasificación: {estado_clf}")
        st.caption("Dataset: 1,500 pólizas · 31 variables")

    # Routing
    if   pagina == " Inicio":                   seccion_inicio(df)
    elif pagina == " Exploración de datos":      seccion_eda(df)
    elif pagina == "️ Preprocesamiento":          seccion_preprocesamiento(df)
    elif pagina == " Modelado":                  seccion_modelado(df, pipe_reg, pipe_clf)
    elif pagina == " Reducción dimensional (PCA)": seccion_pca(df)
    elif pagina == " Simulador actuarial":        seccion_simulador(pipe_reg, pipe_clf)
    elif pagina == " Módulo de imágenes":         seccion_imagenes()
    elif pagina == " Conclusiones":               seccion_conclusiones()


if __name__ == '__main__':
    main()

#fin
    #Gracias por la atención
