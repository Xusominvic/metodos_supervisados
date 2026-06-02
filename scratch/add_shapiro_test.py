import json

notebook_path = "c:\\Users\\Josep\\Documents\\metodos_supervisados\\analisis_explotario.ipynb"

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the Markdown cell
markdown_source = [
    "## 4. Análisis de Normalidad (Test de Shapiro-Wilk)\n",
    "\n",
    "### Teoría: Normalidad y el Test de Shapiro-Wilk\n",
    "En estadística, la suposición de normalidad es un requisito previo para muchos algoritmos paramétricos (como Regresión Lineal, Regresión Logística o Análisis Discriminante Lineal). Si las variables predictoras no siguen una distribución normal, los modelos lineales podrían presentar problemas de rendimiento, y podríamos necesitar aplicar transformaciones (como logarítmicas, raíz cuadrada o Box-Cox) o decantarnos por modelos no paramétricos (como Árboles de Decisión, Random Forest o XGBoost).\n",
    "\n",
    "El **Test de Shapiro-Wilk** (Shapiro y Wilk, 1965) evalúa la hipótesis nula ($H_0$) de que una muestra proviene de una población con distribución normal.\n",
    "- **Estadístico $W$**: Un valor cercano a 1 indica que los datos se ajustan fuertemente a la normalidad. Valores más bajos indican desviaciones.\n",
    "- **P-valor (p-value)**: Si es menor que un nivel de significancia (usualmente $\\alpha = 0.05$), rechazamos $H_0$ y concluimos que los datos **no** son normales. Si es mayor, no podemos rechazar $H_0$.\n",
    "\n",
    "> **Nota Práctica:** El test de Shapiro-Wilk es muy sensible a tamaños de muestra grandes. Con muchos datos (como nuestro caso, N=3000), desviaciones minúsculas de la normalidad pueden dar p-valores muy bajos. Sin embargo, ordenar las variables por el estadístico $W$ nos da una buena métrica relativa de cuáles se desvían más de la campana de Gauss."
]

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": markdown_source
}

# Define the Code cell
code_source = [
    "from scipy.stats import shapiro\n",
    "from typing import Dict, List, Any\n",
    "\n",
    "def run_shapiro_wilk(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:\n",
    "    \"\"\"\n",
    "    Ejecuta el test de Shapiro-Wilk para cada variable numérica en un DataFrame.\n",
    "    \n",
    "    Args:\n",
    "        df: DataFrame de Pandas con las variables numéricas.\n",
    "        alpha: Nivel de significancia para el test.\n",
    "        \n",
    "    Returns:\n",
    "        DataFrame con los resultados (Estadístico W, p-valor y un flag de normalidad).\n",
    "    \"\"\"\n",
    "    results: List[Dict[str, Any]] = []\n",
    "    \n",
    "    # Excluimos la variable objetivo 'class' de este análisis, ya que es categórica\n",
    "    features: pd.DataFrame = df.drop(columns=['class']) if 'class' in df.columns else df\n",
    "    \n",
    "    for col in features.columns:\n",
    "        # SciPy's shapiro lanza un warning si N > 5000, pero nuestro N es 3000\n",
    "        # Calculamos el estadístico y el p-valor\n",
    "        stat, p_value = shapiro(features[col].dropna())\n",
    "        \n",
    "        # Almacenamos los resultados iterativos en un diccionario\n",
    "        results.append({\n",
    "            'Variable': col,\n",
    "            'W_Statistic': stat,\n",
    "            'P_Value': p_value,\n",
    "            'Is_Normal': p_value > alpha\n",
    "        })\n",
    "        \n",
    "    # Convertimos la lista de diccionarios a un DataFrame de Pandas para su visualización\n",
    "    results_df: pd.DataFrame = pd.DataFrame(results)\n",
    "    return results_df\n",
    "\n",
    "# Analizamos todas las variables del conjunto de entrenamiento\n",
    "shapiro_results: pd.DataFrame = run_shapiro_wilk(train_df)\n",
    "\n",
    "print(\"=========================================================\")\n",
    "print(\" Análisis de Normalidad (Todas las Variables)\")\n",
    "print(\"=========================================================\")\n",
    "display(shapiro_results)\n",
    "\n",
    "# Seleccionamos las 10 variables 'más importantes' en términos de desviación de la normalidad.\n",
    "# Ordenando por el estadístico W de menor a mayor, obtenemos las que menos forma normal tienen.\n",
    "top_10_non_normal: pd.DataFrame = shapiro_results.sort_values(by='W_Statistic', ascending=True).head(10)\n",
    "\n",
    "print(\"\\n=========================================================\")\n",
    "print(\" Top 10 Variables con Mayor Desviación de la Normalidad\")\n",
    "print(\" (Ordenadas por menor estadístico W)\")\n",
    "print(\"=========================================================\")\n",
    "display(top_10_non_normal.reset_index(drop=True))"
]

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_source
}

# Append the cells
nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

# Save the notebook back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cells successfully appended to notebook.")
