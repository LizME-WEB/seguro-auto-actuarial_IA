# ============================================================
# Entrenamiento de modelos
# notebooks/01_eda_modelado.ipynb  (sección de modelado)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)
import joblib

# Importar utilidades
# from utils.preprocessing import (
#     preparar_split, construir_preprocesador,
#     NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS, BINARY_COLS,
#     ORDINAL_CATS, TARGET_REG, TARGET_CLF, get_feature_names
# )

# 

# ── Definición de columnas (copia de preprocessing.py) 
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




# Carga y feature engineering 
df = pd.read_csv('data/seguro_auto_actuarial.csv')

df['log_ingreso']          = np.log1p(df['ingreso_mensual_mxn'])
df['edad2']                = df['edad_conductor'] ** 2
df['km_por_vehiculo']      = df['km_anuales'] / (df['edad_vehiculo_anios'] + 1)
df['ratio_prima_suma']     = df['prima_mensual_mxn'] / df['suma_asegurada_mxn'] * 1000
df['tiene_gps']            = df['tiene_gps'].map({'Si':1,'No':0}).astype(float)
df['asistencia_vial']      = df['asistencia_vial'].map({'Si':1,'No':0}).astype(float)
df['mantenimiento_al_dia'] = df['mantenimiento_al_dia'].map(
    {'Si':1,'No':0}).fillna(-1).astype(float)




# Función: construir preprocesador 
def build_preprocessor():
    num = Pipeline([('imp', SimpleImputer(strategy='median')),
                    ('sc', StandardScaler())])
    nom = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore',
                                          sparse_output=False))])
    ord_ = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('enc', OrdinalEncoder(categories=ORDINAL_CATS,
                                            handle_unknown='use_encoded_value',
                                            unknown_value=-1))])
    return ColumnTransformer([
        ('num', num, NUMERIC_COLS),
        ('nom', nom, NOMINAL_COLS),
        ('ord', ord_, ORDINAL_COLS),
        ('bin', 'passthrough', BINARY_COLS),
    ], remainder='drop')




# Función: evalua modelo de regresión
def evaluar_regresion(y_true, y_pred, nombre=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {nombre:<28}  MAE=${mae:>8,.0f}  RMSE=${rmse:>8,.0f}  R²={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}




# Función: evalua modelo de clasificación
def evaluar_clasificacion(y_true, y_pred, nombre=''):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {nombre:<28}  Acc={acc:.3f}  Prec={prec:.3f}  "
          f"Rec={rec:.3f}  F1={f1:.3f}")
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}


# ═════════════
# — REGRESIÓN
# ═════════════
print("=" * 65)
print("REGRESIÓN — costo_esperado_anual_mxn")
print("=" * 65)

X_reg = df.drop(columns=['poliza_id','clase_costo','riesgo_alto', TARGET_REG])
y_reg = df[TARGET_REG]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

print(f"Train: {X_tr.shape[0]} pólizas  |  Test: {X_te.shape[0]} pólizas\n")

#  Modelos lineales 
print("── Modelos lineales ──")
resultados_reg = {}
for nombre, modelo in [
    ('LinearRegression',  LinearRegression()),
    ('Ridge (α=1.0)',     Ridge(alpha=1.0)),
    ('Ridge (α=10.0)',    Ridge(alpha=10.0)),
    ('Lasso (α=1.0)',     Lasso(alpha=1.0, max_iter=5000)),
    ('Lasso (α=50.0)',    Lasso(alpha=50.0, max_iter=5000)),
]:
    pipe = Pipeline([('pre', build_preprocessor()), ('mod', modelo)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    resultados_reg[nombre] = evaluar_regresion(y_te, y_pred, nombre)

# Árboles y ensambles 
print("\n── Árboles y ensambles ──")
for nombre, modelo in [
    ('DTree (max_depth=3)',    DecisionTreeRegressor(max_depth=3, random_state=42)),
    ('DTree (max_depth=6)',    DecisionTreeRegressor(max_depth=6, random_state=42)),
    ('DTree (sin límite)',     DecisionTreeRegressor(max_depth=None, random_state=42)),
    ('RandomForest (150)',     RandomForestRegressor(n_estimators=150,
                                                      random_state=42, n_jobs=-1)),
    ('GradientBoosting',      GradientBoostingRegressor(n_estimators=150,
                                                         random_state=42)),
]:
    pipe = Pipeline([('pre', build_preprocessor()), ('mod', modelo)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    resultados_reg[nombre] = evaluar_regresion(y_te, y_pred, nombre)

# Guardar mejor modelo (Lasso α=50 para interpretabilidad
#        o LinearRegression para simplicidad)
pipe_reg_final = Pipeline([
    ('pre', build_preprocessor()),
    ('mod', Lasso(alpha=50.0, max_iter=5000))
])
pipe_reg_final.fit(X_tr, y_tr)
joblib.dump(pipe_reg_final, 'models/modelo_regresion.joblib')
print("\n✅ Modelo regresión guardado: models/modelo_regresion.joblib")

# Interpretación de coeficientes (Lasso)
pre = pipe_reg_final.named_steps['pre']
ohe_cats = pre.named_transformers_['nom']['ohe'].categories_
nom_names = [f"{c}_{v}" for c, cats in zip(NOMINAL_COLS, ohe_cats) for v in cats]
feat_names = NUMERIC_COLS + nom_names + ORDINAL_COLS + BINARY_COLS
coefs = pipe_reg_final.named_steps['mod'].coef_

# Solo mostrar features activas
activas = [(feat_names[i], coefs[i])
           for i in range(len(coefs)) if coefs[i] != 0]
activas_sorted = sorted(activas, key=lambda x: abs(x[1]), reverse=True)

print(f"\n── Coeficientes Lasso (α=50): {len(activas)} de {len(coefs)} features activas ──")
for nombre, coef in activas_sorted[:10]:
    direccion = "↑" if coef > 0 else "↓"
    print(f"  {nombre:<35}  {direccion} {coef:+,.0f}")

#  Visualización: comparativa de métricas (barras)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
modelos = list(resultados_reg.keys())
colores = ['#185FA5' if 'Linear' in m or 'Ridge' in m or 'Lasso' in m
           else '#3B6D11' for m in modelos]

for ax, metrica, label in zip(axes,
                               ['r2', 'mae', 'rmse'],
                               ['R²', 'MAE (MXN)', 'RMSE (MXN)']):
    vals = [resultados_reg[m][metrica] for m in modelos]
    bars = ax.barh(modelos, vals, color=colores, edgecolor='none')
    ax.set_xlabel(label)
    ax.set_title(label)
    ax.tick_params(labelsize=9)
    # Resaltar mejor
    mejor_idx = (vals.index(max(vals)) if metrica == 'r2'
                 else vals.index(min(vals)))
    bars[mejor_idx].set_edgecolor('#0C447C')
    bars[mejor_idx].set_linewidth(2)

plt.suptitle('Comparativa modelos de regresión — test set', fontsize=12)
plt.tight_layout()
plt.show()



# Feature importance Random Forest Regresión 
pipe_rf_reg = Pipeline([
    ('pre', build_preprocessor()),
    ('mod', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
])
pipe_rf_reg.fit(X_tr, y_tr)

imp_reg = pipe_rf_reg.named_steps['mod'].feature_importances_
pre_rf = pipe_rf_reg.named_steps['pre']
ohe_rf = pre_rf.named_transformers_['nom']['ohe'].categories_
nom_rf = [f"{c}_{v}" for c, cats in zip(NOMINAL_COLS, ohe_rf) for v in cats]
feat_rf = NUMERIC_COLS + nom_rf + ORDINAL_COLS + BINARY_COLS

top_n = 15
top_idx = np.argsort(imp_reg)[-top_n:]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([feat_rf[i] for i in top_idx],
        imp_reg[top_idx], color='#185FA5', edgecolor='none')
ax.set_xlabel('Importancia (Gini impurity decrease)')
ax.set_title(f'Top {top_n} variables — Random Forest Regresión')
plt.tight_layout()
plt.show()


# ═══════════════
# CLASIFICACIÓN

print("\n" + "=" * 65)
print("CLASIFICACIÓN — riesgo_alto")
print("=" * 65)

X_clf = df.drop(columns=['poliza_id','clase_costo','costo_esperado_anual_mxn',
                          TARGET_CLF])
y_clf = df[TARGET_CLF]

# stratify=y_clf preserva la proporción 85/15 en ambos subconjuntos
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

print(f"Train: {Xc_tr.shape[0]} pólizas  "
      f"(riesgo alto: {yc_tr.mean()*100:.1f}%)")
print(f"Test:  {Xc_te.shape[0]} pólizas  "
      f"(riesgo alto: {yc_te.mean()*100:.1f}%)\n")

# Todos los clasificadores 
resultados_clf = {}
pipelines_clf  = {}

for nombre, modelo in [
    ('DTree d=3 balanced',
     DecisionTreeClassifier(max_depth=3, class_weight='balanced',
                             random_state=42)),
    ('DTree d=6 balanced',
     DecisionTreeClassifier(max_depth=6, class_weight='balanced',
                             random_state=42)),
    ('DTree sin límite',
     DecisionTreeClassifier(max_depth=None, random_state=42)),
    ('RF balanced',
     RandomForestClassifier(n_estimators=150, class_weight='balanced',
                             random_state=42, n_jobs=-1)),
    ('RF sin balance',
     RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
    ('GradientBoosting',
     GradientBoostingClassifier(n_estimators=150, random_state=42)),
]:
    pipe = Pipeline([('pre', build_preprocessor()), ('mod', modelo)])
    pipe.fit(Xc_tr, yc_tr)
    y_pred = pipe.predict(Xc_te)
    resultados_clf[nombre] = evaluar_clasificacion(yc_te, y_pred, nombre)
    pipelines_clf[nombre] = pipe

#  Guardar mejor modelo (GradientBoosting — mejor F1) 
joblib.dump(pipelines_clf['GradientBoosting'],
            'models/modelo_clasificacion.joblib')
print("\n✅ Modelo clasificación guardado: models/modelo_clasificacion.joblib")

# Matriz de confusión — GradientBoosting 
y_pred_gb = pipelines_clf['GradientBoosting'].predict(Xc_te)
cm = confusion_matrix(yc_te, y_pred_gb)

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Normal', 'Riesgo Alto'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Matriz de confusión — GradientBoosting\n(mejor F1 = 0.747)')
plt.tight_layout()
plt.show()

#  Visualización: precision vs recall por modelo 
modelos_clf = list(resultados_clf.keys())
precs  = [resultados_clf[m]['prec'] for m in modelos_clf]
recs   = [resultados_clf[m]['rec']  for m in modelos_clf]
f1s    = [resultados_clf[m]['f1']   for m in modelos_clf]

x = np.arange(len(modelos_clf))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width, precs, width, label='Precision', color='#3B6D11',
       alpha=0.85, edgecolor='none')
ax.bar(x,         recs,  width, label='Recall',    color='#185FA5',
       alpha=0.85, edgecolor='none')
ax.bar(x + width, f1s,   width, label='F1',        color='#E24B4A',
       alpha=0.85, edgecolor='none')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(modelos_clf, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Precision / Recall / F1 por modelo — Clasificación riesgo_alto')
ax.legend()
plt.tight_layout()
plt.show()

#  Feature importance — RF Clasificación
pipe_rf_clf = pipelines_clf['RF balanced']
imp_clf = pipe_rf_clf.named_steps['mod'].feature_importances_
pre_rfc = pipe_rf_clf.named_steps['pre']
ohe_rfc = pre_rfc.named_transformers_['nom']['ohe'].categories_
nom_rfc = [f"{c}_{v}" for c, cats in zip(NOMINAL_COLS, ohe_rfc) for v in cats]
feat_rfc = NUMERIC_COLS + nom_rfc + ORDINAL_COLS + BINARY_COLS

top_idx_c = np.argsort(imp_clf)[-top_n:]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([feat_rfc[i] for i in top_idx_c],
        imp_clf[top_idx_c], color='#3B6D11', edgecolor='none')
ax.set_xlabel('Importancia (Gini impurity decrease)')
ax.set_title(f'Top {top_n} variables — RF Clasificación (balanced)')
plt.tight_layout()
plt.show()

#  Análisis del árbol de decisión (árbol pequeño = interpretable) 
from sklearn.tree import export_text

pipe_dt = Pipeline([
    ('pre', build_preprocessor()),
    ('mod', DecisionTreeClassifier(max_depth=3, class_weight='balanced',
                                    random_state=42))
])
pipe_dt.fit(Xc_tr, yc_tr)

# Nombres de features post-transformación
pre_dt  = pipe_dt.named_steps['pre']
ohe_dt  = pre_dt.named_transformers_['nom']['ohe'].categories_
nom_dt  = [f"{c}_{v}" for c, cats in zip(NOMINAL_COLS, ohe_dt) for v in cats]
feat_dt = NUMERIC_COLS + nom_dt + ORDINAL_COLS + BINARY_COLS

dt_model = pipe_dt.named_steps['mod']
reglas = export_text(dt_model,
                     feature_names=feat_dt,
                     max_depth=3)
print("\n── Reglas del árbol de decisión (depth=3) ──")
print(reglas[:2000])  # primeras 2000 chars


print("\n" + "=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
print("REGRESIÓN    → Ganador: LinearRegression / Lasso  (R²=0.43)")
print("CLASIFICACIÓN→ Ganador: GradientBoosting          (F1=0.747)")
print("Modelos guardados en models/")
