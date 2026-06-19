# 
# PCA + Módulo de Imágenes
#

# ╔═══════════════════════════════════════════╗
# ║PCA (para notebook y sección Streamlit)    ║
# ╚═══════════════════════════════════════════╝

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Reutilizar setup
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


def build_preprocessor():
    num  = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler())])
    nom  = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore',
                                           sparse_output=False))])
    ord_ = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('enc', OrdinalEncoder(categories=ORDINAL_CATS,
                                            handle_unknown='use_encoded_value',
                                            unknown_value=-1))])
    return ColumnTransformer([
        ('num', num,  NUMERIC_COLS),
        ('nom', nom,  NOMINAL_COLS),
        ('ord', ord_, ORDINAL_COLS),
        ('bin', 'passthrough', BINARY_COLS),
    ], remainder='drop')


def cargar_y_preparar(path='data/seguro_auto_actuarial.csv'):
    df = pd.read_csv(path)
    df['log_ingreso']          = np.log1p(df['ingreso_mensual_mxn'])
    df['edad2']                = df['edad_conductor'] ** 2
    df['km_por_vehiculo']      = df['km_anuales'] / (df['edad_vehiculo_anios'] + 1)
    df['ratio_prima_suma']     = df['prima_mensual_mxn'] / df['suma_asegurada_mxn'] * 1000
    df['tiene_gps']            = df['tiene_gps'].map({'Si':1,'No':0}).astype(float)
    df['asistencia_vial']      = df['asistencia_vial'].map({'Si':1,'No':0}).astype(float)
    df['mantenimiento_al_dia'] = df['mantenimiento_al_dia'].map(
        {'Si':1,'No':0}).fillna(-1).astype(float)
    return df


# Análisis de varianza explicada 
def analizar_varianza_pca(X_transformed):
    """
    Grafica la varianza explicada acumulada para ayudar a elegir
    el número óptimo de componentes principales.
    """
    pca_full = PCA(random_state=42)
    pca_full.fit(X_transformed)
    var_acum = np.cumsum(pca_full.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Varianza individual por componente (scree plot)
    axes[0].bar(range(1, 21),
                pca_full.explained_variance_ratio_[:20] * 100,
                color='#378ADD', edgecolor='none', alpha=0.85)
    axes[0].set_xlabel('Componente principal')
    axes[0].set_ylabel('Varianza explicada (%)')
    axes[0].set_title('Varianza por componente (primeros 20)')
    axes[0].axhline(5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # ── Varianza acumulada
    axes[1].plot(range(1, len(var_acum)+1), var_acum * 100,
                 color='#185FA5', linewidth=2)
    axes[1].fill_between(range(1, len(var_acum)+1), var_acum * 100,
                          alpha=0.15, color='#185FA5')
    for umbral, color in [(0.80,'#3B6D11'),(0.90,'#EF9F27'),(0.95,'#E24B4A')]:
        n = np.argmax(var_acum >= umbral) + 1
        axes[1].axhline(umbral*100, color=color, linestyle='--',
                        linewidth=1, alpha=0.7,
                        label=f'{umbral*100:.0f}% → {n} componentes')
    axes[1].set_xlabel('Número de componentes')
    axes[1].set_ylabel('Varianza acumulada (%)')
    axes[1].set_title('Varianza explicada acumulada')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(1, len(var_acum))
    axes[1].set_ylim(0, 105)

    plt.suptitle('Análisis de varianza explicada — PCA', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Tabla de umbrales
    print("Componentes necesarios por umbral de varianza:")
    for umbral in [0.80, 0.90, 0.95, 0.99]:
        n = np.argmax(var_acum >= umbral) + 1
        print(f"  {umbral*100:.0f}%  →  {n:2d} componentes")

    return pca_full


#  Visualización 2D coloreada por riesgo_alto 
def visualizar_pca_2d(X_transformed, df_original):
    """
    Proyecta las pólizas en 2 componentes principales y
    las colorea por riesgo_alto y clase_costo.
    """
    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_transformed)

    var1 = pca2.explained_variance_ratio_[0] * 100
    var2 = pca2.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    #  Plot 1: coloreado por riesgo_alto
    colores_riesgo = {0: '#9FE1CB', 1: '#E24B4A'}
    labels_riesgo  = {0: 'Normal (n=1275)', 1: 'Riesgo Alto (n=225)'}

    for clase in [0, 1]:
        mask = df_original['riesgo_alto'].values == clase
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colores_riesgo[clase],
                        alpha=0.4 if clase == 0 else 0.7,
                        s=8 if clase == 0 else 12,
                        label=labels_riesgo[clase],
                        edgecolors='none')

    # Marcar centroides
    for clase in [0, 1]:
        mask = df_original['riesgo_alto'].values == clase
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        axes[0].scatter(cx, cy, c=colores_riesgo[clase],
                        s=120, marker='D', edgecolors='white',
                        linewidths=1.5, zorder=5)
        axes[0].annotate(f'centroide\n({cx:.2f}, {cy:.2f})',
                          (cx, cy), xytext=(10, 10),
                          textcoords='offset points',
                          fontsize=8, color=colores_riesgo[clase])

    axes[0].set_xlabel(f'PC1 ({var1:.1f}% varianza)')
    axes[0].set_ylabel(f'PC2 ({var2:.1f}% varianza)')
    axes[0].set_title('PCA 2D — coloreado por riesgo_alto')
    axes[0].legend(markerscale=2, fontsize=9)
    axes[0].axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    axes[0].axvline(0, color='gray', linewidth=0.5, alpha=0.5)

    # Plot 2: coloreado por clase_costo
    colores_clase = {'Bajo': '#9FE1CB', 'Medio': '#EF9F27', 'Alto': '#E24B4A'}
    for clase, color in colores_clase.items():
        mask = df_original['clase_costo'].values == clase
        n = mask.sum()
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, alpha=0.4, s=8, edgecolors='none',
                        label=f'{clase} (n={n})')

    axes[1].set_xlabel(f'PC1 ({var1:.1f}% varianza)')
    axes[1].set_ylabel(f'PC2 ({var2:.1f}% varianza)')
    axes[1].set_title('PCA 2D — coloreado por clase_costo')
    axes[1].legend(markerscale=2, fontsize=9)
    axes[1].axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5, alpha=0.5)

    plt.suptitle(f'PCA 2D — varianza total capturada: {var1+var2:.1f}%',
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\nVarianza explicada:")
    print(f"  PC1: {var1:.2f}%  |  PC2: {var2:.2f}%  |  Total: {var1+var2:.2f}%")
    print(f"\nCentroide por riesgo_alto:")
    for clase in [0, 1]:
        mask = df_original['riesgo_alto'].values == clase
        print(f"  clase {clase}:  PC1={X_pca[mask,0].mean():.3f}  "
              f"PC2={X_pca[mask,1].mean():.3f}")

    return pca2, X_pca


#  Loadings — qué variables definen cada componente 
def graficar_loadings(pca2, feat_names, n_top=8):
    """
    Muestra las features con mayor contribución a PC1 y PC2.
    """
    components = pca2.components_
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pc_idx, titulo in zip(
            axes, [0, 1], ['PC1 — Loadings', 'PC2 — Loadings']):
        loadings = components[pc_idx]
        top_idx  = np.argsort(np.abs(loadings))[-n_top:]
        top_vals = loadings[top_idx]
        top_lbls = [feat_names[i] for i in top_idx]
        colors   = ['#185FA5' if v > 0 else '#E24B4A' for v in top_vals]

        ax.barh(top_lbls, top_vals, color=colors, edgecolor='none')
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('Loading (contribución al componente)')
        ax.set_title(titulo)
        ax.tick_params(labelsize=9)

    plt.suptitle('Interpretación de los componentes principales', fontsize=12)
    plt.tight_layout()
    plt.show()


#  Ejecutar sección PCA 
if __name__ == '__main__':
    df = cargar_y_preparar()
    X = df.drop(columns=['poliza_id','clase_costo','riesgo_alto',
                          'costo_esperado_anual_mxn'])

    pre = build_preprocessor()
    X_transformed = pre.fit_transform(X)
    print(f"Shape preprocesado: {X_transformed.shape}")

    # Nombres de features
    ohe_cats  = pre.named_transformers_['nom']['ohe'].categories_
    nom_names = [f"{c}_{v}" for c, cats in zip(NOMINAL_COLS, ohe_cats)
                 for v in cats]
    feat_names = NUMERIC_COLS + nom_names + ORDINAL_COLS + BINARY_COLS

    pca_full      = analizar_varianza_pca(X_transformed)
    pca2, X_pca   = visualizar_pca_2d(X_transformed, df)
    graficar_loadings(pca2, feat_names)


# ╔═════════════════════════════════════════
# ║ MÓDULO DE IMÁGENES (Sección Streamlit) ║
# ╚═════════════════════════════════════════



def seccion_imagenes_streamlit():
    """
    Función lista para insertar en app.py de Streamlit.
    Copia este bloque dentro del if/elif de tu navegación.
    """
    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    st.header(" Módulo de análisis de imágenes")
    st.markdown(
        "Carga una fotografía de un **vehículo** o **evidencia de siniestro** "
        "para explorar transformaciones de procesamiento de imágenes con OpenCV."
    )

    #  Carga de imagen ─
    archivo = st.file_uploader(
        "Sube una imagen (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"],
        help="Sube una foto de vehículo o evidencia de siniestro"
    )

    if archivo is None:
        st.info("Sube una imagen para comenzar. Puedes usar cualquier foto "
                "de un automóvil.")
        return

    # array OpenCV
    pil_img  = Image.open(archivo).convert('RGB')
    img_rgb  = np.array(pil_img)
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h_orig, w_orig = img_rgb.shape[:2]

    # ── Parámetros interactivos en sidebar 
    st.sidebar.markdown("### ️ Parámetros de transformación")
    resize_w    = st.sidebar.slider("Resize — ancho (px)", 64, 512, 256, 32)
    resize_h    = st.sidebar.slider("Resize — alto (px)",  64, 512, 192, 32)
    blur_kernel = st.sidebar.slider("Blur — tamaño del kernel", 3, 31, 11, 2)
    crop_pct    = st.sidebar.slider("Crop — % desde arriba",    0, 40,  20,  5)
    sharp_int   = st.sidebar.slider("Sharpening — intensidad",  1,  5,  2,   1)

    # ──  transformaciones 

    # 1. Resize
    img_resized = cv2.resize(img_rgb, (resize_w, resize_h))

    # 2. Crop — recortar la región inferior (zona del vehículo)
    y_start = int(h_orig * crop_pct / 100)
    img_crop = img_rgb[y_start:, :]

    # 3. Blur gaussiano
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    img_blur = cv2.GaussianBlur(img_rgb, (k, k), 0)

    # 4. Sharpening con kernel personalizado
    sharp_kernel = np.array([
        [ 0,            -sharp_int,            0],
        [-sharp_int,  4*sharp_int+1, -sharp_int],
        [ 0,            -sharp_int,            0]
    ])
    img_sharp = cv2.filter2D(img_rgb, -1, sharp_kernel)
    img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)

    # 5. Contraste — histogram equalization en canal Y (YUV)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_contrast = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # 6. Detección de bordes (Canny)
    median_int    = np.median(img_gray)
    lower_thresh  = int(max(0,   (1.0 - 0.33) * median_int))
    upper_thresh  = int(min(255, (1.0 + 0.33) * median_int))
    img_edges     = cv2.Canny(img_gray, lower_thresh, upper_thresh)

    

    # ── Grid de imágenes 
    st.subheader("Transformaciones aplicadas")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_rgb,      caption=f"Original ({w_orig}×{h_orig}px)",
                 use_column_width=True)
        st.image(img_blur,     caption=f"Blur Gaussiano (kernel {k}×{k})",
                 use_column_width=True)

    with col2:
        st.image(img_resized,  caption=f"Resize → {resize_w}×{resize_h}px",
                 use_column_width=True)
        st.image(img_sharp,    caption=f"Sharpening (intensidad {sharp_int})",
                 use_column_width=True)

    with col3:
        st.image(img_crop,     caption=f"Crop (recorte desde {crop_pct}%)",
                 use_column_width=True)
        st.image(img_contrast, caption="Contraste (Histogram Equalization)",
                 use_column_width=True)

    # Bordes en fila completa
    st.image(img_edges,
             caption=(f"Detección de bordes — Canny "
                      f"(low={lower_thresh}, high={upper_thresh})"),
             use_column_width=True,
             channels="GRAY")




    # ── Histograma de color 
    st.subheader("Histograma de distribución de color (RGB)")
    st.markdown(
        "Muestra cuántos píxeles hay de cada intensidad (0–255) por canal. "
        "Útil para detectar sobreexposición o subexposición en fotos de siniestro."
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    colores_canal = [('#E24B4A', 'Rojo'), ('#3B6D11', 'Verde'), ('#185FA5', 'Azul')]
    for i, (color, nombre) in enumerate(colores_canal):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, linewidth=1.2, alpha=0.85, label=nombre)

    ax.set_xlim([0, 256])
    ax.set_xlabel('Intensidad del píxel (0=negro, 255=blanco)')
    ax.set_ylabel('Número de píxeles')
    ax.set_title('Histograma de color — imagen cargada')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # ── Aislamiento de color por rango HSV 
    st.subheader("Aislamiento de color (rango HSV)")
    col_a, col_b = st.columns(2)
    with col_a:
        h_low  = st.slider("Matiz (Hue) mínimo",  0, 180,  90)
        s_low  = st.slider("Saturación mínima",   0, 255, 100)
        v_low  = st.slider("Valor mínimo",         0, 255,  50)
    with col_b:
        h_high = st.slider("Matiz (Hue) máximo",  0, 180, 130)
        s_high = st.slider("Saturación máxima",   0, 255, 255)
        v_high = st.slider("Valor máximo",         0, 255, 255)

    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mascara  = cv2.inRange(img_hsv,
                            np.array([h_low, s_low, v_low]),
                            np.array([h_high, s_high, v_high]))
    img_mask = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.image(mascara, caption="Máscara (blanco = rango seleccionado)",
                 channels="GRAY", use_column_width=True)
    with col_m2:
        st.image(img_mask, caption="Imagen con color aislado",
                 use_column_width=True)

    # ── Estadísticas de la imagen 
    st.subheader("Estadísticas de la imagen")
    canales_stats = pd.DataFrame({
        'Canal': ['Rojo', 'Verde', 'Azul'],
        'Media': [img_rgb[:,:,i].mean().round(1) for i in range(3)],
        'Desv. Std': [img_rgb[:,:,i].std().round(1) for i in range(3)],
        'Mínimo': [img_rgb[:,:,i].min() for i in range(3)],
        'Máximo': [img_rgb[:,:,i].max() for i in range(3)],
    })
    st.dataframe(canales_stats, hide_index=True, use_container_width=True)



    # Explicación actuarial
    with st.expander(" ¿Para qué sirve esto en un sistema de seguros?"):
        st.markdown("""
        En un sistema real de seguros de automóvil, el procesamiento de imágenes permite:

        - **Evaluación de daños**: Aislar zonas dañadas del vehículo mediante detección
          de bordes y cambios de color.
        - **Verificación de estado**: Comparar el estado del vehículo al contratar
          vs al reportar un siniestro.
        - **Detección de anomalías**: La detección de bordes (Canny) puede revelar
          deformaciones estructurales no visibles a simple vista.
        - **Clasificación de severidad**: Un histograma desplazado hacia valores bajos
          puede indicar una foto nocturna de un siniestro grave.

        *Este módulo demuestra las herramientas básicas del capítulo 8 del libro.*
        """)
