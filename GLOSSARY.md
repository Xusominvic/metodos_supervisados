# Glosario Científico de Aprendizaje Supervisado

Este documento recopila y define de forma precisa los términos técnicos, algoritmos y métricas más importantes utilizados durante la implementación y evaluación de esta práctica.

---

## Métricas de Evaluación

### 1. Accuracy (Exactitud / Porcentaje de Acierto)
Métrica global de clasificación que mide la proporción de predicciones correctas sobre el total de observaciones:
$$Accuracy = \frac{VP + VN}{VP + VN + FP + FN}$$
* **Nota técnica:** Es muy engañosa cuando las clases están desbalanceadas, ya que no discrimina si el error ocurre en clases mayoritarias o minoritarias.

### 2. Balanced Accuracy (Exactitud Balanceada)
Métrica de clasificación adaptada a datos desbalanceados. Se calcula como el promedio del recall (sensibilidad) obtenido en cada una de las clases:
$$Balanced\ Accuracy = \frac{Recall_0 + Recall_1 + \dots + Recall_C}{C}$$
* **Nota técnica:** Evita estimaciones optimistas cuando un modelo ignora una clase minoritaria. Un modelo que prediga aleatoriamente obtendrá un valor de $1/C$ (en nuestro caso, 33.3%).

### 3. F1-Score (Medida F1)
Media armónica entre la Precisión (proporción de predicciones positivas correctas) y el Recall (proporción de positivos reales identificados):
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
* **F1-Macro (Macro promedio):** Calcula el F1-Score de forma independiente para cada clase y luego calcula su promedio aritmético. Otorga el mismo peso a todas las clases, independientemente de su tamaño.

---

## Estrategias y Técnicas

### 4. Búsqueda en Rejilla (GridSearchCV)
Método sistemático de ajuste de hiperparámetros que evalúa exhaustivamente todas las combinaciones posibles de una cuadrícula definida, estimando el rendimiento de cada combinación mediante Validación Cruzada.

### 5. Validación Cruzada Estratificada (Stratified K-Fold CV)
Técnica de partición de datos para estimar la capacidad de generalización de un modelo. Divide el conjunto de datos en $K$ pliegues (folds), garantizando que la proporción de clases en cada pliegue de entrenamiento y validación sea aproximadamente igual a la del conjunto de datos completo.
* **Nota técnica:** Es obligatoria cuando existen clases extremadamente minoritarias para asegurar que cada pliegue contenga suficientes muestras de todas las clases.

### 6. Aprendizaje Sensible al Costo (Cost-Sensitive Learning)
Enfoque de entrenamiento que asocia diferentes penalizaciones (pesos) a los errores de clasificación según la clase. La técnica `class_weight='balanced'` ajusta la función de pérdida del optimizador ponderando cada muestra de forma inversamente proporcional a la frecuencia de su clase en el conjunto de entrenamiento.

### 7. Fuga de Datos (Data Leakage)
Error metodológico donde información del conjunto de validación o prueba "se filtra" al conjunto de entrenamiento durante pasos de preprocesamiento (como el escalado de características o selección de variables). Esto sesga los resultados y produce métricas de validación artificialmente altas que no se mantendrán en datos reales de prueba.

---

## Modelos y Algoritmos

### 8. Random Forest Classifier
Algoritmo de ensamble que combina múltiples árboles de decisión entrenados en paralelo sobre diferentes subconjuntos de datos (mediante *Bootstrap Aggregation* o Bagging) y seleccionando un subconjunto aleatorio de variables en cada división de nodo (*Feature Subspacing*). Ayuda a reducir la varianza del modelo.

### 9. Gradient Boosting Classifier
Algoritmo de ensamble que construye árboles de decisión de forma secuencial. Cada nuevo árbol intenta predecir los residuos (errores) del modelo anterior en la dirección del gradiente de la función de pérdida. Controlado por el `learning_rate` (tasa de aprendizaje) y la profundidad de los árboles individuales.

### 10. AdaBoost Classifier
Algoritmo de boosting secuencial que ajusta modelos débiles (usualmente árboles de decisión con profundidad 1 o "stumps") sobre versiones modificadas de los datos, donde los pesos de las observaciones mal clasificadas se incrementan en cada iteración para forzar al siguiente estimador a centrarse en los casos difíciles.

### 11. Support Vector Machine (SVM) con Kernel RBF
Clasificador que busca encontrar el hiperplano óptimo de separación maximizando el margen entre clases. Al usar un kernel de Función de Base Radial (RBF), mapea implícitamente los datos a un espacio de características de dimensión infinita para encontrar fronteras de decisión no lineales en el espacio original.

### 12. Bagging Classifier
Algoritmo de ensamble genérico que entrena múltiples estimadores base en paralelo sobre muestras de entrenamiento generadas aleatoriamente con reemplazo (bootstrap). Reduce la varianza del estimador base sin alterar sustancialmente su sesgo.
