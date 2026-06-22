# DEVELOPMENT LOG

## [2026-04-20] - Inicialización del Proyecto
### Cambios:
- Creación de la estructura base del proyecto.
- Configuración de los archivos de documentación obligatorios (`PROJECT_OVERVIEW.md` y `DEVELOPMENT_LOG.md`).
- Inicio del notebook `analisis_explotario.ipynb` para la carga inicial de datos.

## [2026-04-20] - Análisis de Balanceo de Clases
### Cambios:
- Añadida sección de análisis de la variable objetivo `class` en el notebook de EDA.
- Implementación de cálculos de frecuencia absoluta y relativa.
- Visualización mediante `countplot` para inspección visual rápida.

### Lógica Detrás del Cambio:
- El balanceo de clases es un paso crítico en el Análisis Exploratorio de Datos. Permite determinar si será necesario aplicar técnicas de remuestreo (SMOTE, undersampling) o ajustar las métricas de evaluación (evitando el Accuracy puro en favor de Precision-Recall).

### Impacto Esperado:
- Identificación temprana de sesgos en el dataset.
- Información clave para la posterior elección de funciones de pérdida o pesos en los modelos supervisados.

## [2026-04-20] - Corrección de Dependencias (Seaborn)
### Cambios:
- Eliminación de la dependencia de `seaborn` en el notebook de EDA.
- Refactorización de las visualizaciones para utilizar únicamente `matplotlib.pyplot`.
- Actualización del Stack Técnico en `PROJECT_OVERVIEW.md`.

### Lógica Detrás del Cambio:
- Se identificó un `ModuleNotFoundError` al intentar importar `seaborn`. Tras inspeccionar el entorno, se confirmó que solo `matplotlib` está disponible. Para no bloquear el avance del estudiante, se optó por una implementación más robusta basada en la librería base.

### Impacto Esperado:
- El notebook ahora es ejecutable sin necesidad de instalaciones adicionales en el entorno actual.

### Lógica Detrás del Cambio:
- Cumplimiento de los estándares académicos del Máster para asegurar la trazabilidad y reproducibilidad desde el inicio.
- Preparación del entorno para realizar un Análisis Exploratorio de Datos (EDA) robusto.

## [2026-04-24] - Mejora en Visualización de Correlaciones
### Cambios:
- Se ha modificado la visualización de los pares de variables con alta correlación (`high_corr`) para mostrar la lista completa sin truncamiento.
- Uso de `pd.option_context('display.max_rows', None)` para asegurar que todas las filas sean visibles en el notebook.

### Lógica Detrás del Cambio:
- La inspección de la multicolinealidad es fundamental para la selección de variables. Ver solo los primeros 5-8 pares puede ocultar relaciones importantes entre variables que podrían afectar la estabilidad de los coeficientes en modelos lineales o el rendimiento en modelos basados en distancias.

### Impacto Esperado:
- Visibilidad total sobre las variables redundantes o altamente correlacionadas, facilitando la toma de decisiones sobre reducción de dimensionalidad o eliminación de variables.

## [2026-04-24] - Inclusión de Análisis de Varianza Cero (Variables Constantes)
### Cambios:
- Se ha añadido una nueva sección al final del notebook (`analisis_explotario.ipynb`) para identificar variables con varianza cero y varianza muy baja (< 0.01).
- Inclusión de una celda explicativa (Markdown) detallando la teoría de las variables constantes y la maldición de la dimensionalidad.

### Lógica Detrás del Cambio:
- Las variables constantes no aportan información discriminativa al modelo y consumen recursos computacionales. A nivel académico y práctico, este chequeo debe formar parte de todo Análisis Exploratorio de Datos (EDA) para asegurar la calidad de las características (features) que se alimentan a los algoritmos supervisados.
- Se incluyó la identificación de varianzas muy bajas para detectar variables cuasi-constantes que podrían aportar más ruido que señal.

### Impacto Esperado:
- Mejora en la estabilidad del entrenamiento posterior y reducción de la dimensionalidad de los datos.
- Prevención de problemas matemáticos en el modelado predictivo o escalado de datos (e.g. división por cero).

## [2026-04-24] - Refactorización de Distribución de Variables (Sección 5)
### Cambios:
- Se ha modificado el código de la "Sección 5. Distribución de Variables" en `analisis_explotario.ipynb`.
- En lugar de iterar y graficar las distribuciones de todas las variables predictoras del dataset, el código ahora:
  1. Calcula dinámicamente las correlaciones de Pearson con la variable objetivo (`class`).
  2. Selecciona las 5 variables con la mayor correlación absoluta.
  3. Genera los histogramas y curvas KDE únicamente para este subconjunto.

### Lógica Detrás del Cambio:
- En datasets con un número moderado a alto de variables (como este con 50 predictores), intentar visualizar la distribución de todas ellas simultáneamente produce exceso de ruido visual (Information Overload).
- Al centrarnos en el top 5 de variables con más *poder predictivo lineal*, enfocamos el Análisis Exploratorio en las características que más probabilidades tienen de discriminar a la clase objetivo, facilitando la identificación visual de patrones o separabilidad.

### Impacto Esperado:
- Gráficos mucho más legibles y útiles para la interpretación directa del usuario y del reporte académico.
- Ahorro de tiempo computacional al evitar renderizar decenas de subgráficos irrelevantes.

## [2026-04-24] - Análisis de Normalidad (Test de Shapiro-Wilk)
### Cambios:
- Inclusión de una nueva sección en el EDA (`analisis_explotario.ipynb`) para evaluar la normalidad de las variables predictoras.
- Implementación de la prueba estadística de Shapiro-Wilk mediante `scipy.stats.shapiro`.
- Generación de tabla con resultados completos (estadístico W, p-valor y diagnóstico) y una tabla resumen extrayendo las 10 variables con mayor desviación de la normalidad.

### Lógica Detrás del Cambio:
- De acuerdo con el rigor científico y el enfoque de Primeros Principios exigido en el Máster, la comprobación distribucional es obligatoria. Validar la asunción de normalidad determinará si podemos emplear modelos paramétricos tradicionales o si es imperativo transformar los datos o decantarse por modelos no paramétricos.
- Dado el tamaño muestral (N=3000), el test es extremadamente sensible, por lo que organizar las variables por su estadístico W resulta ser una forma más pragmática de encontrar las mayores anomalías distributivas.

### Impacto Esperado:
- Justificación formal (matemática y estadística) de futuras transformaciones de variables (Box-Cox, logaritmo, etc.) y/o selección del tipo de modelos de Machine Learning en las siguientes fases del proyecto.

## [2026-04-24] - Implementación de PCA y Escalado
### Cambios:
- Creación de un nuevo archivo `pca.ipynb`.
- Implementación de un pipeline que incluye estandarización de los datos (`StandardScaler`).
- Aplicación de la \"Gráfica del Codo\" (Scree Plot) para evaluar la varianza explicada acumulada sin limitar el número de componentes.
- Aplicación de Análisis de Componentes Principales (`PCA`) para reducción de dimensionalidad a 2 componentes.
- Inclusión de gráfico de dispersión mostrando la proyección de las dos componentes coloreadas por clase.

### Lógica Detrás del Cambio:
- **Primeros Principios:** PCA es una técnica de aprendizaje no supervisado útil para reducción de dimensionalidad. Funciona maximizando la varianza proyectada. Al depender de la escala y la varianza, es mandatorio estandarizar (`StandardScaler`) los datos para que variables con rangos mayores no dominen las componentes principales. 
- Analizar la varianza acumulada mediante la **Gráfica del Codo** permite decidir de forma cuantitativa y objetiva con cuántas componentes nos debemos quedar en función de la pérdida de información aceptable (90% o 95%).
- Reducir a 2 dimensiones permite una visualización geométrica directa que nos ayuda a intuir si las clases son separables linealmente antes de aplicar algoritmos más complejos.

### Impacto Esperado:
- Obtención de una métrica clara sobre la dimensionalidad real intrínseca de los datos (mediante la curva de varianza acumulada).
- Obtención de una intuición visual sobre la estructura de los datos y separabilidad de los grupos (clases).
- Base preparada para una posible inclusión de PCA como paso de preprocesamiento dentro de pipelines predictivos en caso de que la alta dimensionalidad genere ruido o altos costes de computación.

## [2026-06-16] - Optimización de Hiperparámetros con GridSearchCV y Estrategias Mixtas de Balanceo
### Cambios:
- Creación del script `grid_search_optimization.py` con 5 GridSearchCV independientes para Random Forest, Gradient Boosting, AdaBoost, SVM (RBF) y Bagging.
- Implementación de estrategia mixta de balanceo:
  - **class_weight='balanced'** para modelos que lo soportan nativamente (Random Forest, SVM, Bagging).
  - **SMOTE** (Chawla et al., 2002) integrado en `imblearn.pipeline.Pipeline` para modelos secuenciales (Gradient Boosting, AdaBoost), garantizando que el sobremuestreo se aplica SOLO dentro de cada fold de CV (sin data leakage).
- Configuración estándar: `cv=5` (StratifiedKFold), `scoring='accuracy'`, `n_jobs=-1`.

### Lógica Detrás del Cambio:
- La versión anterior del notebook (`modelos_supervisados.ipynb`) usaba parámetros por defecto y `sample_weight` manual para GradientBoosting. Esta evolución busca encontrar la configuración óptima de cada modelo mediante búsqueda exhaustiva en rejilla.
- Se separa en un script `.py` para facilitar la ejecución desde el notebook y la reutilización del código.
- El uso de `ImbPipeline` de imblearn (en vez del Pipeline de sklearn) es crítico: sklearn.Pipeline no soporta transformadores que modifiquen tanto X como y (como hace SMOTE).

### Impacto Esperado:
- Mejora significativa en el rendimiento de los modelos al explorar combinaciones de hiperparámetros.
- Resultados reproducibles gracias a `random_state=42` en todos los componentes.
- Base para la comparativa final de modelos y selección del mejor clasificador.

## [2026-06-22] - v3: Optimización para Máxima Accuracy Global (sin balanceo artificial)
### Cambios:
- **Eliminado SMOTE** de todos los pipelines. Ya no se usa `imblearn.pipeline.Pipeline`, se vuelve a `sklearn.pipeline.Pipeline`.
- **Eliminado `class_weight='balanced'`** de Random Forest, SVM y Bagging.
- **Cambiada la métrica** de optimización de `f1_macro` a `accuracy`.
- **Mantenido `SelectKBest(f_classif)`** como filtro de irrelevancia, con rango ampliado a `k ∈ [10, 15, 20, 25, 35, 45]`.
- Añadido reporte de **desviación estándar** del score en CV para cada modelo.

### Lógica Detrás del Cambio:
- Los resultados de v2 mostraron F1-macro muy pobres (0.37–0.45) con overfitting extremo (brechas Train-CV de hasta +0.63).
- **Causa raíz**: las estrategias de balanceo (SMOTE + class_weight) penalizaban la accuracy al sacrificar predicciones correctas en las clases 0 y 1 (99.4% de los datos) para intentar detectar la Clase 2 (0.6%).
- El enunciado del trabajo exige **maximizar la accuracy global**, por lo que forzar la detección de la clase minoritaria era contraproducente.
- Se mantiene `SelectKBest` porque en v2 demostró empíricamente que los modelos rendían mejor con 10-25 variables que con las 45 completas, lo que indica presencia de variables irrelevantes (no redundantes, que ya fueron eliminadas en el EDA).

### Impacto Esperado:
- Mejora significativa en accuracy al no distorsionar las fronteras de decisión con balanceo artificial.
- Reducción del overfitting al eliminar las muestras sintéticas de SMOTE.
- La Clase 2 probablemente será ignorada por los modelos, pero esto es consistente con maximizar accuracy global.
