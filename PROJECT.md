# Guía del Proyecto: Decisiones Metodológicas y Científicas

Este documento contiene la justificación formal y científica de cada decisión tomada en la preparación, entrenamiento y evaluación de modelos supervisados para esta práctica.

---

## 1. Tratamiento del Desbalanceo Extremo de Clases

### Contexto de los Datos
* **Clase 0:** 68.7% (Mayoritaria)
* **Clase 1:** 30.7% (Intermedia)
* **Clase 2:** 0.6% (Minoritaria extrema, 19 de 3000 observaciones)

### Justificación de la Estrategia Dual (Opción 2)
Para responder con rigor al enunciado sin arriesgar la calificación del test, diseñamos dos configuraciones:

#### A. Enfoque Estándar (Maximización de la Exactitud Global)
* **Método:** Entrenamiento convencional sin pesos en la pérdida ni remuestreo. Optimización en `GridSearchCV` guiada por la métrica de `accuracy`.
* **Sustento Científico:** En problemas donde el objetivo de rendimiento (función de utilidad o nota académica) es puramente la tasa de acierto global ($Accuracy = \frac{VP + VN}{N}$), ignorar la clase minoritaria extrema reduce el riesgo de cometer falsos positivos en las clases mayoritarias. Esto es óptimo en términos estadísticos globales cuando la distribución del conjunto de test se asume idéntica a la de entrenamiento.

#### B. Enfoque Balanceado (Cost-Sensitive Learning)
* **Método:** Uso de `class_weight='balanced'` en clasificadores paramétricos y basados en árboles (Random Forest, SVM, Bagging) y optimización en `GridSearchCV` guiada por `balanced_accuracy`.
* **Sustento Científico:** La asignación de pesos inversamente proporcionales a las frecuencias de las clases ($w_c = \frac{N}{C \cdot n_c}$) penaliza fuertemente los errores en la clase minoritaria. Esto ajusta el umbral de decisión del modelo hacia las clases mayoritarias, permitiendo detectar observaciones de la Clase 2 (mayor *Recall*). Usar `balanced_accuracy` (media aritmética del recall por clase) evita el sesgo optimista que produce la exactitud tradicional sobre datasets desbalanceados.

---

## 2. Prevención de Fugas de Información (Data Leakage) mediante Pipelines

### Justificación del uso de `sklearn.pipeline.Pipeline`
* **Decisión:** Integrar el escalado (`StandardScaler`), la selección univariada de características (`SelectKBest`) y el clasificador dentro de un único pipeline de scikit-learn.
* **Sustento Científico:** Si el escalado o la selección de características se aplicaran a todo el conjunto de entrenamiento *antes* de dividir en pliegues de validación cruzada (Cross-Validation), las estadísticas globales (media, desviación estándar, coeficientes F de ANOVA) influirían en el entrenamiento de cada pliegue. Esto se conoce como **fuga de datos (data leakage)** y conduce a estimaciones de generalización optimistas pero falsas. Al usar un `Pipeline`, el ajuste (`fit`) de los transformadores ocurre estrictamente dentro del conjunto de entrenamiento de cada pliegue, garantizando que el pliegue de validación permanezca completamente "fuera de muestra" (out-of-sample).

---

## 3. Selección de Características Univariada (`SelectKBest`)

### Justificación
* **Decisión:** Añadir un selector de características univariado basado en el test ANOVA F (`f_classif`) y buscar el hiperparámetro `k` optimizándolo en la rejilla.
* **Sustento Científico:** El EDA reveló variables con varianza casi nula (< 0.01) y fuerte colinealidad. Los clasificadores basados en distancias (como SVM con kernel RBF) sufren la "maldición de la dimensionalidad", donde variables ruidosas o irrelevantes distorsionan las distancias euclídeas. Al introducir `SelectKBest`, evaluamos individualmente el poder discriminativo lineal de cada característica frente a la variable objetivo, filtrando variables irrelevantes antes del entrenamiento del clasificador.

---

## 4. Normalización mediante `StandardScaler` (Solo SVM)

### Justificación
* **Decisión:** Aplicar estandarización (media = 0, desviación estándar = 1) **únicamente** en el pipeline de SVM (RBF). Los modelos basados en árboles (Random Forest, Gradient Boosting, AdaBoost, Bagging) **no llevan escalado**.
* **Sustento Científico:** Los árboles de decisión y sus ensambles operan mediante particiones binarias basadas en umbrales sobre cada variable individual. Estas particiones son invariantes a transformaciones monótonas (como el escalado lineal), porque reordenar o reescalar los valores no cambia el punto de corte óptimo. Por el contrario, SVM con kernel RBF calcula distancias euclídeas en un espacio de alta dimensión: $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$. Si las variables tienen escalas muy distintas, las de mayor magnitud dominarán la norma $\|x_i - x_j\|$ y distorsionarán la frontera de decisión.

---

## 5. Eliminación de Variables Redundantes (Multicolinealidad)

### Justificación
* **Decisión:** Eliminar las variables `['bpsmt', 'csuhz', 'gwrec', 'glhls', 'bqwyz']` antes del entrenamiento. Estas 5 columnas fueron identificadas en el EDA como pares con correlación de Pearson muy alta entre sí.
* **Sustento Científico:** Cuando dos variables están altamente correlacionadas ($|r| > 0.8$), ambas aportan información casi idéntica al modelo. Mantener ambas incrementa la dimensionalidad sin añadir poder discriminativo, lo cual:
  - En SVM, infla artificialmente la norma del vector de características y diluye la contribución de variables realmente informativas.
  - En modelos de ensamble basados en árboles, la selección aleatoria de variables (`max_features`) puede seleccionar repetidamente variables redundantes en lugar de variables independientes, reduciendo la diversidad real entre los árboles del ensamble.
  - En `SelectKBest(f_classif)`, dos variables colineales pueden competir por las mismas posiciones del ranking, desplazando a variables con información complementaria.
* **Resultado:** Pasamos de 50 a 45 features, reduciendo el espacio de búsqueda y mejorando la señal útil por dimensión.

---

## 6. Validación de Desplazamiento de Distribución (Dataset/Label Shift) mediante Proyección PCA

### Justificación
* **Decisión:** Ajustar un PCA 2D sobre el conjunto de entrenamiento estandarizado y proyectar tanto `X_train` como `X_test` para generar una gráfica comparativa antes de tomar la decisión sobre qué enfoque de entrenamiento priorizar.
* **Sustento Científico:** En competiciones o tareas académicas con conjuntos de test ocultos, existe el riesgo latente de **Dataset Shift** (donde $P(X_{train}) \neq P(X_{test})$) o **Label Shift** (donde $P(y_{train}) \neq P(y_{test})$). Si el conjunto de test tuviera una distribución de clases drásticamente diferente (por ejemplo, mayor prevalencia de la Clase 2 minoritaria para penalizar clasificadores sesgados), los datos de test se proyectarían de forma concentrada sobre las regiones geométricas del espacio latente correspondientes a dicha clase en train. Proyectar ambos conjuntos sobre los mismos autovectores de PCA nos permite contrastar visual y estadísticamente si las densidades marginales coinciden, dotando a la selección del modelo final de evidencia empírica de generalización en lugar de basarse puramente en supuestos.

---

## 7. Resultados Empíricos y Selección Final del Modelo

Tras ejecutar las búsquedas en rejilla (`GridSearchCV` con 5-fold CV) evaluando un total de ~3000 ajustes, el análisis de los resultados revela comportamientos clave en los distintos clasificadores:

### 7.1. Análisis de Overfitting (Varianza)
* **Modelos de Alta Varianza:** Random Forest y Bagging presentan un claro sobreajuste. Alcanzan una precisión en entrenamiento del 100% (`Train Accuracy = 1.00`), pero su generalización en validación cruzada cae drásticamente (Brecha de overfitting $\approx 0.26$).
* **Modelos Robustos:** Gradient Boosting, SVM y AdaBoost muestran brechas mucho más estrechas, indicando una mejor capacidad de generalización. AdaBoost es el menos propenso al sobreajuste (brecha $\approx 0.017$), aunque su rendimiento base es inferior.

### 7.2. El Trade-off del Desbalanceo (Accuracy vs Recall Minoritario)
* **Enfoque Estándar:** Maximiza la `CV Accuracy` global (hasta un **0.7477** con Gradient Boosting). Sin embargo, al observar el desglose por clases, modelos como Random Forest y Bagging ignoran por completo la Clase 2 (Recall = 0%).
* **Enfoque Balanceado:** Al introducir pesos inversamente proporcionales a la frecuencia de clase (`class_weight='balanced'`), se sacrifica una mínima fracción de exactitud global (Gradient Boosting baja a **0.7450**) a cambio de triplicar o cuadruplicar el Recall de las clases minoritarias, mejorando sustancialmente el `Balanced Accuracy` y el `Macro F1-Score`.

### 7.3. Selección Final
De acuerdo con las directrices típicas donde la métrica principal a optimizar es la exactitud global (Accuracy) salvo que se especifique lo contrario:
* **Modelo Seleccionado:** **Gradient Boosting (Enfoque Estándar)**
* **Hiperparámetros Óptimos:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`, `feature_selection__k=45`.
* **Rendimiento Esperado:** CV Accuracy = **0.7477** $\pm$ 0.0279.
* **Predicción:** Este es el modelo reentrenado con el 100% de los datos de `training.csv` que se ha utilizado para generar las etiquetas finales en `test_completado.csv`.

