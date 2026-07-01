# Métodos Supervisados: Trabajo Práctico

**Autor:** Josep
**Curso:** Aprendizaje Automático / Minería de Datos
**Fecha:** Julio 2026

---

## 1. Introducción

El presente trabajo aborda un problema de clasificación supervisada multi-clase a partir de un conjunto de datos sintéticos completamente anonimizados (`training.csv` y `test.csv`). La variable objetivo (`class`) presenta tres niveles (0, 1 y 2), y se caracteriza por poseer un **desbalanceo extremo**: la Clase 0 representa el 68.7% de la muestra, la Clase 1 el 30.7%, y la Clase 2 un marginal 0.6%.

El enfoque general adoptado para resolver el reto ha sido altamente sistemático y automatizado. En lugar de afinamientos manuales aislados, se ha construido un entorno de evaluación basado en la clase `Pipeline` de *scikit-learn*, integrando la selección de características (`SelectKBest`) y el propio estimador. Todo ello se ha optimizado simultáneamente mediante una búsqueda exhaustiva en rejilla (`GridSearchCV`) con validación cruzada de 5 particiones (5-fold CV). 

Además, dado el riesgo que supone el desbalanceo, el análisis se bifurcó en dos metodologías competitivas:
1.  **Enfoque Estándar:** Optimización pura de la exactitud global (`accuracy`).
2.  **Enfoque Balanceado:** Optimización de la exactitud por clases (`balanced_accuracy`) compensando los pesos probabilísticos (`class_weight`).

Los resultados globales demuestran que el modelo de **Gradient Boosting** (bajo el enfoque estándar) es el más competitivo, logrando una `CV Accuracy` del **0.7477** (74.77%), manteniendo una robustez envidiable frente al sobreajuste si lo comparamos con alternativas basadas en árboles paralelos como Random Forest o Bagging.

---

## 2. Método

La estrategia de modelado exigía garantizar que la validación fuera justa y que ningún algoritmo tuviera ventajas injustas por fugas de datos (Data Leakage). Por ello, todos los procesos (reducción de dimensionalidad, estandarización y entrenamiento) se encapsularon en *Pipelines*.

### 2.1. Preprocesamiento y Análisis de Componentes (PCA)
1.  **Limpieza de Colinealidad:** Mediante un análisis previo de correlación (Spearman), se identificaron pares de variables perfectamente redundantes ($|\rho| \approx 1$). Se eliminaron variables como `bpsmt`, `csuhz`, `gwrec`, `glhls` y `bqwyz` para evitar inestabilidades en los cálculos matriciales.
2.  **Validación de Desplazamiento (Dataset Shift):** Al disponer de un conjunto de test oculto sin etiquetas, se proyectaron tanto `X_train` como `X_test` en un subespacio de 2 componentes principales (PCA). Observamos que las distribuciones conjuntas se superponen perfectamente, descartando la presencia de *Dataset Shift* o *Label Shift* inducido artificialmente en el conjunto de prueba.
3.  **Estandarización Selectiva:** Solo el modelo Support Vector Machine (SVM) requiere imperativamente que las variables compartan escala (dado que usa distancias euclídeas subyacentes). Para SVM se incluyó un `StandardScaler` en su Pipeline. Los modelos basados en reglas o árboles (RF, Bagging, GB, AdaBoost) no se escalaron, ya que son invariantes a transformaciones monótonas de las variables.
4.  **Selección de Características:** Se integró `SelectKBest` con el test estadístico ANOVA (`f_classif`) en cada Pipeline. Esto filtra el ruido en alta dimensionalidad antes de pasarlo al estimador.

### 2.2. Estrategia por Clasificador
Para cada uno de los 5 algoritmos seleccionados, se definió un espacio de búsqueda (`param_grid`) justificado empíricamente:

*   **Random Forest:** Se exploró el número de estimadores, el límite de profundidad (`max_depth`) y la fracción de características (`max_features`). Tiene sentido explorar profundidades infinitas (`None`) vs podas (`10`) para controlar su altísima varianza teórica.
*   **Gradient Boosting:** Tratando de equilibrar el *learning_rate* (0.01, 0.05, 0.1) con la profundidad máxima (3, 4). Es vital mantener la profundidad baja para que los aprendices débiles no sobreajusten el ruido.
*   **AdaBoost:** Se evaluaron distintas tasas de aprendizaje (0.1, 0.5, 1.0) sobre el estimador base por defecto (tocones de profundidad 1).
*   **Support Vector Machine (RBF):** Para su kernel no lineal se exploró la constante de regularización $C$ (1, 10, 100) y la apertura matricial $\gamma$ (0.1, 0.01, 0.001) para equilibrar la flexibilidad de la frontera de decisión.
*   **Bagging:** Basado en árboles de decisión, se probaron distintos números de estimadores y fracciones de muestreo de *Bootstrap* (`max_samples` al 0.8 y 1.0).

---

## 3. Resultados

A continuación, se detalla la tabla resumen exigida con los hiperparámetros óptimos que maximizaron la exactitud (`Accuracy`) en el 5-fold CV bajo el esquema Estándar.

| Classifier | Best parameters | Train accuracy | 5-fold CV accuracy |
| :--- | :--- | :---: | :---: |
| GradientBoostingClassifier | learning_rate=0.05, max_depth=3, n_estimators=200, k=45 | 0.8843 | 0.7477 |
| SVC (RBF Kernel) | C=10, gamma=0.01, k=15 | 0.7783 | 0.7420 |
| BaggingClassifier | max_samples=1.0, n_estimators=150, k=45 | 1.0000 | 0.7377 |
| RandomForestClassifier | max_depth=None, max_features='sqrt', n_estimators=300, k=25 | 1.0000 | 0.7367 |
| AdaBoostClassifier | learning_rate=0.1, n_estimators=200, k=35 | 0.7330 | 0.7157 |

*(Nota: "k" hace referencia al número de características óptimas seleccionadas por `SelectKBest` en el Pipeline).*

### 3.1. Evaluación del Tamaño y Complejidad
Podemos observar que los modelos de ensamble tienden a requerir un número moderadamente alto de modelos base para converger (entre 150 y 300 árboles). Sorprendentemente, `SelectKBest` determinó que retener entre 35 y 45 variables (de las 52 originales) es lo idóneo para Gradient Boosting y Bagging, lo que sugiere que aproximadamente 10 variables del dataset original solo aportan ruido estadístico puro. Sin embargo, para SVM y Random Forest, la criba fue más agresiva (k=15 y k=25), confiando solo en el núcleo más predictivo del dataset.

---

## 4. Discusión

El experimento arrojó lecciones críticas relativas al equilibrio entre sesgo, varianza, y el peligro de los datos fuertemente desbalanceados.

### 4.1. El problema del Sobreajuste (Brecha de Varianza)
Los modelos con estructura profunda de árbol paralelizables (Random Forest sin poda explícita y Bagging) exhiben un memorismo absoluto (`Train Accuracy = 1.0000`). Esto genera una penalización teórica enorme frente a datos nuevos: su brecha entre entrenamiento y validación cruzada es superior a 0.26. Por el contrario, los ensamblados secuenciales (Gradient Boosting y AdaBoost) controlan estructuralmente la varianza porque sus árboles base están obligatoriamente limitados en profundidad (`max_depth=3` o tocones), reportando brechas de overfitting muchísimo menores (0.13 y 0.01 respectivamente).

### 4.2. El Trade-off del Desbalanceo
Para validar las limitaciones de los clasificadores ante la Clase 2 (solo 4 casos en un *fold* promedio), contrastamos la ejecución Estándar frente a una Balanceada (`class_weight='balanced'`). Los hallazgos guían claramente la discusión:

*   **La ceguera de la Accuracy:** Bajo optimización estándar, Random Forest, AdaBoost, SVM y Bagging fallan estrepitosamente en detectar un solo caso de la Clase 2 (`Recall = 0.0%`). Les resulta estadísticamente más rentable fallar sistemáticamente el 0.6% de las veces que arriesgarse a cometer falsos positivos sobre el 68.7% mayoritario.
*   **Sensibilidad Forzada:** Al forzar el balanceo, algoritmos matemáticamente robustos como **SVM** despiertan y disparan su tasa de acierto minoritario del 0% al **21%**. Sin embargo, la limitación inherente a este enfoque es el peaje en "Falsos Positivos": al volverse hipersensible a la clase 2, SVM destroza su exactitud global bajando bruscamente del 0.74 al 0.65.
*   **Gradient Boosting como Punto Óptimo:** Gradient Boosting es el único algoritmo que, incluso bajo el entorno estándar, logra "ver" a la minoría (identifica ~15.8% de la Clase 2), dominando simultáneamente la exactitud de las clases mayoritarias.

Teniendo en cuenta que el test de PCA dictaminó que la distribución geométrica de testeo es análoga a la de entrenamiento, forzar una equidad artificial en los pesos solo aumentaría los falsos positivos en el momento de la inferencia final. La optimización de la `Accuracy` cruda es metodológicamente coherente con los datos.

---

## 5. Conclusión

Este trabajo iterativo permitió cribar características correlacionadas, ajustar la complejidad matricial, y evaluar empíricamente cinco filosofías de Machine Learning.

Se concluye que las arquitecturas paralelizables (Bagging/RF) son excesivamente propensas al sobreajuste en distribuciones ruidosas, mientras que SVM es excesivamente lábil frente a los hiperparámetros de penalización de clase. 

El modelo que ofrece la superioridad empírica combinada es el **Gradient Boosting**. Su naturaleza secuencial corrige los errores de los árboles precursores de forma robusta, alcanzando un **$74.77\%$ de acierto validado**. Este modelo, tras ser reentrenado con el 100% de la muestra de `training.csv`, ha sido el seleccionado para inferir y completar los resultados en `test.csv`.

Como mejoras futuras, se podría implementar `SMOTE` (Synthetic Minority Over-sampling Technique) en la base de datos de entrenamiento del Pipeline, para dotar de densidad geométrica a la Clase 2 sin destrozar la estabilidad matemática inherente al escalado de pesos que penalizó a la SVM en este estudio.
