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

### Impacto Esperado:
- Estructura clara para el desarrollo posterior de modelos supervisados.
- Facilitación de la carga de datos para los siguientes pasos de preprocesamiento.
