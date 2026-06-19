# ============================================================
# EDA — seguro_auto_actuarial.csv
# Proyecto: Sistema actuarial de predicción de riesgo
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/seguro_auto_actuarial.csv')



#  1. ESTRUCTURA GENERAL 
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
print(df.dtypes.to_string())



# 2. VALORES FALTANTES 
faltantes = df.isnull().sum()
faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
pct = (faltantes / len(df) * 100).round(1)
resumen_nan = pd.DataFrame({'n_faltantes': faltantes, 'pct': pct})
print("\n", resumen_nan)

# Visualización de faltantes
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(faltantes.index, pct.values, color='#E24B4A', edgecolor='none')
ax.set_xlabel('% de valores faltantes')
ax.set_title('Columnas con valores faltantes')
for i, v in enumerate(pct.values):
    ax.text(v + 0.1, i, f'{v}%', va='center', fontsize=10)
plt.tight_layout()
plt.show()



#  3. OUTLIERS con IQR 
num_cols = ['edad_conductor', 'antiguedad_cliente_anios', 'ingreso_mensual_mxn',
            'score_crediticio', 'prima_mensual_mxn', 'suma_asegurada_mxn',
            'km_anuales', 'edad_vehiculo_anios', 'costo_esperado_anual_mxn']

def detectar_outliers_iqr(df, cols):
    resultados = {}
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        resultados[col] = {
            'n_outliers': n_out,
            'pct': round(n_out / len(df) * 100, 2),
            'lim_inf': round(lo, 1),
            'lim_sup': round(hi, 1)
        }
    return pd.DataFrame(resultados).T.sort_values('pct', ascending=False)

print("\n", detectar_outliers_iqr(df, num_cols))



# 4. DISTRIBUCIONES + SESGO 
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
cols_plot = ['costo_esperado_anual_mxn', 'ingreso_mensual_mxn', 'prima_mensual_mxn',
             'suma_asegurada_mxn', 'km_anuales', 'score_crediticio']

for ax, col in zip(axes.flatten(), cols_plot):
    ax.hist(df[col].dropna(), bins=40, color='#378ADD', edgecolor='none', alpha=0.8)
    skew = df[col].skew()
    color = '#E24B4A' if abs(skew) > 2 else ('#EF9F27' if abs(skew) > 0.5 else '#3B6D11')
    ax.set_title(f'{col}\nskew = {skew:.2f}', fontsize=10, color=color)
    ax.tick_params(labelsize=9)

plt.suptitle('Distribuciones de variables numéricas — detectar sesgo', fontsize=12, y=1.01)
plt.tight_layout()
plt.show()



# 5. BOXPLOTS (outliers visuales)
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col in zip(axes, ['ingreso_mensual_mxn', 'prima_mensual_mxn',
                            'suma_asegurada_mxn', 'costo_esperado_anual_mxn']):
    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor='#B5D4F4', color='#185FA5'),
               medianprops=dict(color='#0C447C', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='#E24B4A',
                               markersize=4, alpha=0.5))
    ax.set_title(col.replace('_mxn', '\n(MXN)'), fontsize=9)
    ax.tick_params(labelsize=8)
plt.suptitle('Boxplots — distribución y outliers', fontsize=11)
plt.tight_layout()
plt.show()



# 6. BALANCE DE CLASES
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# riesgo_alto
vc = df['riesgo_alto'].value_counts()
axes[0].bar(['Normal (0)', 'Riesgo alto (1)'], vc.values,
            color=['#9FE1CB', '#E24B4A'], edgecolor='none')
axes[0].set_title('Balance de clases: riesgo_alto', fontsize=11)
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 10, f'{v}\n({v/len(df)*100:.1f}%)',
                 ha='center', fontsize=10)
axes[0].set_ylim(0, 1400)

# clase_costo
vc2 = df['clase_costo'].value_counts()
colors_costo = {'Bajo': '#9FE1CB', 'Medio': '#EF9F27', 'Alto': '#E24B4A'}
axes[1].bar(vc2.index, vc2.values,
            color=[colors_costo.get(k, '#378ADD') for k in vc2.index],
            edgecolor='none')
axes[1].set_title('Distribución: clase_costo', fontsize=11)
for i, (lbl, v) in enumerate(vc2.items()):
    axes[1].text(i, v + 10, f'{v}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Nota: clase_costo NO debe usarse como feature de entrada (data leakage)
# Es solo para visualizaciones y validación del target



#  7. CORRELACIONES 
num_all = ['edad_conductor', 'antiguedad_cliente_anios', 'ingreso_mensual_mxn',
           'score_crediticio', 'prima_mensual_mxn', 'suma_asegurada_mxn',
           'deducible_pct', 'historial_siniestros_3_anios', 'km_anuales',
           'edad_vehiculo_anios', 'puntaje_riesgo_zona', 'numero_siniestros_12m',
           'costo_esperado_anual_mxn', 'riesgo_alto']

corr = df[num_all].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.4, cbar_kws={'shrink': 0.8},
            annot_kws={'fontsize': 9})
ax.set_title('Matriz de correlaciones — variables numéricas', fontsize=12, pad=12)
plt.tight_layout()
plt.show()



#  8. CATEGÓRICAS vs TARGET 
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Fila 1: vs costo mediano
for ax, col in zip(axes[0], ['tipo_vehiculo', 'uso_vehiculo', 'zona_residencia']):
    orden = (df.groupby(col)['costo_esperado_anual_mxn']
               .median().sort_values(ascending=False))
    ax.bar(orden.index, orden.values, color='#378ADD', edgecolor='none')
    ax.set_title(f'{col} vs costo mediano', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# Fila 2: vs tasa de riesgo_alto
for ax, col in zip(axes[1], ['tipo_vehiculo', 'uso_vehiculo', 'canal_venta']):
    tasa = (df.groupby(col)['riesgo_alto']
              .mean().sort_values(ascending=False) * 100)
    ax.bar(tasa.index, tasa.values, color='#E24B4A', edgecolor='none', alpha=0.8)
    ax.axhline(15, color='gray', linestyle='--', linewidth=1, label='Media global (15%)')
    ax.set_title(f'{col} vs % riesgo alto', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.set_ylabel('%')
    ax.legend(fontsize=8)

plt.suptitle('Variables categóricas vs targets', fontsize=12)
plt.tight_layout()
plt.show()



#  9. EDAD DEL CONDUCTOR — efecto no lineal 
df['rango_edad'] = pd.cut(df['edad_conductor'],
                           bins=[17, 25, 35, 50, 65, 76],
                           labels=['18-25', '26-35', '36-50', '51-65', '66+'])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
costo_edad = df.groupby('rango_edad', observed=True)['costo_esperado_anual_mxn'].mean()
axes[0].bar(costo_edad.index, costo_edad.values, color='#378ADD', edgecolor='none')
axes[0].set_title('Edad del conductor vs costo esperado promedio', fontsize=10)
axes[0].set_ylabel('Costo promedio (MXN)')
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

riesgo_edad = df.groupby('rango_edad', observed=True)['riesgo_alto'].mean() * 100
axes[1].bar(riesgo_edad.index, riesgo_edad.values, color='#E24B4A', edgecolor='none')
axes[1].axhline(15, color='gray', linestyle='--', linewidth=1)
axes[1].set_title('Edad del conductor vs % riesgo alto', fontsize=10)
axes[1].set_ylabel('% riesgo alto')

plt.suptitle('Efecto no lineal de la edad — curva en U', fontsize=11)
plt.tight_layout()
plt.show()

# Limpiar columna temporal
df.drop(columns=['rango_edad'], inplace=True)

print("\n EDA completo.")
print("Conclusiones clave:")
print("  1. costo e ingreso tienen sesgo extremo (skew>5) → aplicar log(x)")
print("  2. prima_mensual tiene 9.9% de outliers genuinos → RobustScaler o log")
print("  3. Faltantes ≤7.1% en todas las columnas → imputar, no eliminar filas")
print("  4. riesgo_alto desbalanceado (85/15) → class_weight='balanced'")
print("  5. suma_asegurada es el predictor lineal más fuerte (r=0.62)")
print("  6. edad tiene efecto no lineal (U) → añadir feature edad²")
