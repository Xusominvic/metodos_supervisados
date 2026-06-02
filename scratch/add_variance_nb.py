import json

def add_variance_cells(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Comprobar si ya existe la seccion
    for cell in data['cells']:
        if cell['cell_type'] == 'markdown':
            if 'Análisis de Varianza Cero' in "".join(cell['source']):
                print("La sección de varianza ya existe en el notebook.")
                return

    markdown_cell = {
        
        "cell_type": "markdown",
        "id": "variance-theory",
        "metadata": {},
        "source": [
            "## 7. Análisis de Varianza Cero y Baja (Constantes y Cuasi-Constantes)\n",
            "\n",
            "### Teoría: Variables con Varianza Cero\n",
            "Las variables con varianza cero tienen exactamente el mismo valor para todas las observaciones del dataset. En los algoritmos supervisados, estas características no aportan ninguna información útil para discriminar entre las distintas clases (o predecir un valor numérico continuo), pues no muestran variación alguna.\n",
            "\n",
            "Mantener este tipo de variables puede ocasionar problemas matemáticos (como divisiones por cero al estandarizar o calcular correlaciones) y aumenta innecesariamente la dimensionalidad (agravando la *maldición de la dimensionalidad*). Por esta razón, el procedimiento estándar es identificarlas y eliminarlas.\n",
            "\n",
            "Adicionalmente, es útil observar las variables con varianza **muy baja** (cuasi-constantes), ya que en la mayoría de los casos actúan prácticamente como una constante y aportan más ruido que señal."
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "variance-analysis",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cálculo de la varianza para todas las variables predictoras numéricas\n",
            "feature_cols = [c for c in train_df.select_dtypes(include='number').columns if c != 'class']\n",
            "variances = train_df[feature_cols].var()\n",
            "\n",
            "# Identificar variables con varianza cero\n",
            "zero_variance_cols = variances[variances == 0].index.tolist()\n",
            "\n",
            "print(f\"Número de variables con varianza cero en train: {len(zero_variance_cols)}\")\n",
            "if len(zero_variance_cols) > 0:\n",
            "    print(f\"Variables con varianza cero (candidatas a ser eliminadas): {zero_variance_cols}\")\n",
            "else:\n",
            "    print(\"✅ No se encontraron variables predictoras con varianza cero en el conjunto de entrenamiento.\")\n",
            "\n",
            "# Identificar variables con varianza muy baja (cuasi-constantes)\n",
            "umbral_varianza = 0.01\n",
            "low_variance_cols = variances[(variances > 0) & (variances < umbral_varianza)].sort_values()\n",
            "print(f\"\\nNúmero de variables con varianza muy baja (< {umbral_varianza}) en train: {len(low_variance_cols)}\")\n",
            "if len(low_variance_cols) > 0:\n",
            "    print(\"Variables con varianza muy baja:\")\n",
            "    display(low_variance_cols.to_frame(name='Varianza'))"
        ]
    }
    
    # We remove empty cells at the end of the notebook if any
    while len(data['cells']) > 0 and data['cells'][-1]['cell_type'] == 'code' and not data['cells'][-1]['source']:
        data['cells'].pop()

    data['cells'].append(markdown_cell)
    data['cells'].append(code_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully with variance analysis.")

if __name__ == "__main__":
    add_variance_cells('analisis_explotario.ipynb')
