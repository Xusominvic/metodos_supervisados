# PROJECT OVERVIEW

## 1. Naturaleza del Proyecto
Este proyecto forma parte de la asignatura de **Aprendizaje Automático** del Máster en IA para Investigación. El objetivo es aplicar métodos supervisados para resolver un problema de clasificación basado en los datos proporcionados en `training.csv` y `test.csv`.

## 2. Stack Técnico
- **Lenguaje:** Python 3.x
- **Librerías Disponibles/Principales:**
    - `pandas`: Manipulación y análisis de datos.
    - `numpy`: Cálculos numéricos.
    - `matplotlib`: Visualización de datos (usada como base para evitar dependencias externas faltantes).
    - `jupyter`: Entorno de desarrollo interactivo.
    *Nota: Se ha detectado la ausencia de `seaborn` y `scikit-learn` en el entorno base, por lo que se priorizará el uso de librerías estándar o se solicitará su instalación.*

## 3. Arquitectura del Proyecto
- `data/`: Contiene los conjuntos de datos en formato CSV.
    - `training.csv`: Datos de entrenamiento con etiquetas.
    - `test.csv`: Datos de prueba.
- `doc/`: Documentación oficial y enunciados.
    - `enunciado.pdf`: Descripción detallada del trabajo.
- `analisis_explotario.ipynb`: Notebook dedicado al Análisis Exploratorio de Datos (EDA).
- `DEVELOPMENT_LOG.md`: Trazabilidad de cambios y decisiones técnicas.
