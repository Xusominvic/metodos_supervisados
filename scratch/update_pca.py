import json

filepath = r"c:\Users\Josep\Documents\metodos_supervisados\pca.ipynb"

with open(filepath, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

new_markdown_cell = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Determinación del Número Óptimo de Componentes (Gráfica del Codo)\n",
    "\n",
    "Antes de aplicar la transformación final para visualización 2D, utilizamos la \"gráfica del codo\" (Scree Plot) y la curva de varianza acumulada para determinar cuántas componentes principales son necesarias para retener una cantidad aceptable de información (por ejemplo, el 90% o 95% de la varianza original).\n",
    "Para ello, calcularemos el PCA sobre todas las características estandarizadas y analizaremos la contribución de cada componente."
   ]
}

new_code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Estandarizamos todas las características para la gráfica de varianza\n",
    "scaler_test = StandardScaler()\n",
    "X_scaled_test = scaler_test.fit_transform(X_features)\n",
    "\n",
    "# Aplicamos PCA sin limitar el número de componentes\n",
    "pca_test = PCA(random_state=RANDOM_SEED)\n",
    "pca_test.fit(X_scaled_test)\n",
    "\n",
    "# Calculamos la varianza explicada acumulada\n",
    "cumulative_variance = np.cumsum(pca_test.explained_variance_ratio_)\n",
    "\n",
    "# Generamos la gráfica del codo\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b', markersize=4)\n",
    "plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.8, label='Umbral 90% de Varianza')\n",
    "plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.8, label='Umbral 95% de Varianza')\n",
    "plt.title('Análisis de Varianza Explicada Acumulada (Scree Plot)', fontsize=14, pad=15)\n",
    "plt.xlabel('Número de Componentes Principales', fontsize=12)\n",
    "plt.ylabel('Varianza Explicada Acumulada', fontsize=12)\n",
    "plt.legend(loc='lower right')\n",
    "plt.grid(True, linestyle='--', alpha=0.6)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
}

# Insert the new cells after cell 1
notebook['cells'].insert(2, new_code_cell)
notebook['cells'].insert(2, new_markdown_cell)

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Updated pca.ipynb successfully")
