# Análisis Comparativo de Coste-Beneficio (Estándar vs Balanceado)

Este análisis desglosa el *trade-off* exacto que ocurre al obligar a los algoritmos a prestar atención a la clase minoritaria (Clase 2: 0.6% de la muestra). Comparamos las métricas empíricas obtenidas en la validación cruzada para cada modelo.

---

### Random Forest

| Métrica | Enfoque Estándar | Enfoque Balanceado | Variación (Impacto) |
| :--- | :---: | :---: | :---: |
| **CV Accuracy (Global)** | 0.7367 | 0.7210 | -0.0157 |
| **CV Balanced Accuracy** | 0.4166 | 0.4072 | -0.0094 |
| **Recall Clase 2 (Difícil)** | 0.0000 (0.0%) | 0.0000 (0.0%) | +0.0000 (+0.0%) |

**Interpretación del Trade-off:**
Este algoritmo, incluso forzándolo con pesos balanceados, es incapaz de detectar la clase minoritaria (Recall 0%). El desbalanceo es demasiado severo para su estructura de hiperparámetros óptima, lo que hace que la Balanced Accuracy caiga drásticamente en ambos casos (acercándose al 0.33, equivalente a un clasificador aleatorio entre 3 clases).

---

### Gradient Boosting

| Métrica | Enfoque Estándar | Enfoque Balanceado | Variación (Impacto) |
| :--- | :---: | :---: | :---: |
| **CV Accuracy (Global)** | 0.7477 | 0.7450 | -0.0027 |
| **CV Balanced Accuracy** | 0.4841 | 0.4908 | +0.0067 |
| **Recall Clase 2 (Difícil)** | 0.1579 (15.8%) | 0.1579 (15.8%) | +0.0000 (+0.0%) |

**Interpretación del Trade-off:**
En este caso particular, el modelo mantiene su nivel de detección en la clase 2 constante en ambos enfoques (es el único que logra detectarla sin ayuda explícita). Sin embargo, los pesos alterados en el enfoque balanceado han hecho que mejore su acierto en la clase 1 (clase media), subiendo su Balanced Accuracy global en +0.0067. Todo esto a cambio de una ínfima caída en exactitud general (-0.27%).

---

### AdaBoost

| Métrica | Enfoque Estándar | Enfoque Balanceado | Variación (Impacto) |
| :--- | :---: | :---: | :---: |
| **CV Accuracy (Global)** | 0.7157 | 0.6667 | -0.0490 |
| **CV Balanced Accuracy** | 0.3766 | 0.4070 | +0.0304 |
| **Recall Clase 2 (Difícil)** | 0.0000 (0.0%) | 0.0526 (5.3%) | +0.0526 (+5.3%) |

**Interpretación del Trade-off:**
Al aplicar el enfoque balanceado, el modelo "despierta" y consigue detectar un 5.3% de los casos de la clase minoritaria (antes ignorados). El "precio a pagar" por esta mejora en equidad es una caída de casi un 5% en la exactitud global (CV Accuracy). La ganancia neta en justicia de clase se refleja en un aumento de +0.0304 en la Balanced Accuracy.

---

### SVM (RBF)

| Métrica | Enfoque Estándar | Enfoque Balanceado | Variación (Impacto) |
| :--- | :---: | :---: | :---: |
| **CV Accuracy (Global)** | 0.7420 | 0.6513 | -0.0907 |
| **CV Balanced Accuracy** | 0.4266 | 0.4884 | +0.0618 |
| **Recall Clase 2 (Difícil)** | 0.0000 (0.0%) | 0.2105 (21.1%) | +0.2105 (+21.1%) |

**Interpretación del Trade-off:**
Este es el ejemplo de libro sobre el "Coste-Beneficio" del desbalanceo. En el modo estándar ignora a la minoría (Recall 0%). Al usar `class_weight='balanced'`, SVM es el algoritmo que matemáticamente mejor responde: logra detectar un extraordinario **21.1%** de la clase minoritaria. Sin embargo, su hipersensibilidad a esta clase provoca que cometa "falsos positivos" en el resto de clases, lo que penaliza la exactitud global hundiéndola un 9% (cae del 0.74 al 0.65). 

---

### Bagging

| Métrica | Enfoque Estándar | Enfoque Balanceado | Variación (Impacto) |
| :--- | :---: | :---: | :---: |
| **CV Accuracy (Global)** | 0.7377 | 0.7073 | -0.0304 |
| **CV Balanced Accuracy** | 0.4259 | 0.3828 | -0.0431 |
| **Recall Clase 2 (Difícil)** | 0.0000 (0.0%) | 0.0000 (0.0%) | +0.0000 (+0.0%) |

**Interpretación del Trade-off:**
Igual que Random Forest, Bagging basado en árboles complejos no reacciona positivamente al balanceo; pierde exactitud global y sigue siendo ciego frente a la Clase 2.

---

## Conclusión Ejecutiva del Análisis

El análisis demuestra numéricamente el clásico dilema de la clasificación altamente desbalanceada:

1. **Los Modelos Ciegos:** Algoritmos como Random Forest o Bagging, optimizados pura y duramente para Accuracy global, consiguen valores respetables (~73.7%) a base de sacrificar por completo a la minoría (Recall de 0%).
2. **El "Despertar" de SVM:** Al introducir pesos de balanceo, SVM es el modelo que mejor obedece matemáticamente y pasa de ignorar la clase 2 (0%) a liderar su identificación (21%). El coste asociado es severo (la exactitud global cae al 65%), lo que significa que el modelo se ha vuelto un poco "paranoico" prediciendo la clase 2 en casos que no lo eran (Falsos Positivos).
3. **El Punto Dulce (Gradient Boosting):** Es el claro ganador empírico del experimento. Es el único algoritmo que, gracias a su ensamblado iterativo secuencial de corrección de errores, consigue detectar casos minoritarios (15.79%) incluso en su versión estándar sin ayudas, y a la vez logra mantener la mejor Accuracy global de todos (74.77%).
