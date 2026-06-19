#  Sistema Actuarial de Predicción de Riesgo en Seguros de Automóvil

**Proyecto Integrador · Actuaría y Ciencia de Datos**  
Profesor:
Dr. José Alberto Guzmán Torres

Integrantes:
Lizzet Mendoza Esteban
Sofía Libertad Flores Ochoa
Ana Berenice García Hernández

## Descripción

Aplicación en Streamlit que integra limpieza de datos, feature engineering,
modelos de regresión y clasificación, PCA y procesamiento de imágenes
sobre una cartera sintética de 1,500 pólizas de seguro de automóvil.

**Variables objetivo:**
- `costo_esperado_anual_mxn` — regresión (Lasso / LinearRegression)
- `riesgo_alto` — clasificación binaria (GradientBoosting, clase desbalanceada 15%)



## Instalación

```bash
# 1. Descomprimir el proyecto
cd proyecto_seguro_actuarial

en esta sección ocupamos mucho la terminal de la computadora, ya que trabajamos en MacOs, siempre trabajabamos en windows, solo que se nos descompuso la compu chida :.(

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```



## Ejecución

```bash
streamlit run app.py
```
pero obvio, tenemos que estar en dentro de proyecto_seguro_actuarial, o donde este el proyecto.

La app abre automáticamente en `http://localhost:8501`

al inicio pide un correo electrónico, pero al momento de ingresarlo, ya te manda a la página directamente.
Pero de ahí no pone trabas, se los juro.

## Entrenar modelos, despues de tantos intentos xd

Los modelos se guardan en `models/`. 


Desde el notebook:

```bash
jupyter notebook notebooks/01_eda_modelado.ipynb
```



## Estructura del proyecto, esto recomendado por el profe jiji

```
proyecto_seguro_actuarial/
├── app.py                          ← App Streamlit principal
├── requirements.txt
├── README.md
├── data/
│   └── seguro_auto_actuarial.csv   ← Base de datos sintética
├── notebooks/
│   └── 01_eda_modelado.ipynb       ← EDA y modelado completo
├── models/
│   ├── modelo_regresion.joblib     ← Pipeline regresión (Lasso)
│   └── modelo_clasificacion.joblib ← Pipeline clasificación (GradientBoosting)
├── utils/
│   ├── preprocessing.py            ← Feature engineering y Pipeline
│   └── plots.py                    ← Funciones de graficación reutilizables
└── assets/
    └── imagenes_ejemplo_vehiculos/ ← Imágenes de ejemplo para el módulo
```



## Secciones de la app

| Sección | Contenido |
|---|---|
|  Inicio | Contexto actuarial, métricas del dataset, diccionario |
|  Exploración | Faltantes, distribuciones, correlaciones, balance de clases |
|  Preprocesamiento | Feature engineering, Pipeline, ColumnTransformer |
|  Modelado | Comparativa de modelos, métricas, matriz de confusión |
|  PCA | Scree plot, visualización 2D, loadings |
|  Simulador | Formulario para predicción de nueva póliza |
|  Imágenes | Transformaciones OpenCV sobre foto de vehículo |
|  Conclusiones | Hallazgos, limitaciones, ética, mejoras futuras |


## Decisiones

- **Preprocesamiento**: Pipeline de sklearn con ColumnTransformer para evitar data leakage
- **Regresión**: Lasso(α=50) — mejor MAE y selección automática de features
- **Clasificación**: GradientBoosting — mejor F1=0.747, equilibrio precision/recall
- **Desbalance**: evaluado con precision, recall, F1 y matriz de confusión
- **PCA 2D**: captura 16.12% de varianza — útil para visualización exploratoria
- **Features engineeradas**: log_ingreso, edad², km_por_vehiculo, ratio_prima_suma

---

## Principales

- Python 3.10+ (tratamos de actualizar a la versión no más actual, sino, 3.11.. o 3.12 máximo, si descargabamos una más actual, nos ponía varias trabas )
- streamlit, pandas, numpy, scikit-learn (estos esenciales, realizamos anotaciones de los capítulos y sus videos)
- matplotlib, seaborn
- opencv-python-headless, Pillow
