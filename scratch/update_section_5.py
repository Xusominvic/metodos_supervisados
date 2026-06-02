import json

def update_section_5(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_source = [
        "# == 5. DISTRIBUCION DE LAS 5 VARIABLES MAS CORRELACIONADAS ===================\n",
        "from scipy.stats import gaussian_kde\n",
        "import math\n",
        "\n",
        "# 1. Calcular la correlacion de todas las variables con 'class'\n",
        "corr_with_class = train_df.corr(method='pearson')['class'].drop('class').dropna()\n",
        "\n",
        "# 2. Seleccionar las 5 variables con mayor correlacion (en valor absoluto)\n",
        "top_5_vars = corr_with_class.abs().sort_values(ascending=False).head(5).index.tolist()\n",
        "print(f\"Las 5 variables con mayor correlacion absoluta con la clase son: {top_5_vars}\")\n",
        "\n",
        "N_COLS = min(5, len(top_5_vars))\n",
        "N_ROWS = math.ceil(len(top_5_vars) / N_COLS)\n",
        "\n",
        "fig, axes = plt.subplots(\n",
        "    N_ROWS, N_COLS,\n",
        "    figsize=(N_COLS * 4.5, N_ROWS * 4),\n",
        "    constrained_layout=True,\n",
        "    squeeze=False\n",
        ")\n",
        "axes_flat = axes.flatten()\n",
        "\n",
        "# Paleta por clase para comparar distribuciones\n",
        "palette = {0: '#4C72B0', 1: '#DD8452'}\n",
        "clases  = sorted(train_df['class'].dropna().unique())\n",
        "\n",
        "for i, col in enumerate(top_5_vars):\n",
        "    ax = axes_flat[i]\n",
        "    for cls in clases:\n",
        "        datos = train_df.loc[train_df['class'] == cls, col].dropna()\n",
        "        color = palette.get(cls, 'grey')\n",
        "\n",
        "        # Histograma normalizado\n",
        "        ax.hist(datos, bins=30, color=color, alpha=0.35, density=True)\n",
        "\n",
        "        # Curva KDE\n",
        "        if len(datos) > 1 and datos.std() > 0:\n",
        "            kde   = gaussian_kde(datos, bw_method='scott')\n",
        "            x_rng = np.linspace(datos.min(), datos.max(), 200)\n",
        "            ax.plot(x_rng, kde(x_rng), color=color, linewidth=1.8,\n",
        "                    label=f'Clase {int(cls)}')\n",
        "\n",
        "    ax.set_title(f'{col} (corr: {corr_with_class[col]:.3f})', fontsize=11, fontweight='bold', pad=6)\n",
        "    ax.set_xlabel('')\n",
        "    ax.set_ylabel('Densidad', fontsize=9)\n",
        "    ax.tick_params(labelsize=8)\n",
        "    ax.grid(axis='y', linestyle='--', alpha=0.4)\n",
        "    if i == 0:\n",
        "        ax.legend(fontsize=9, loc='upper right')\n",
        "\n",
        "# Ocultar ejes sobrantes (si los hubiera)\n",
        "for ax in axes_flat[len(top_5_vars):]:\n",
        "    ax.set_visible(False)\n",
        "\n",
        "fig.suptitle(\n",
        "    'Distribucion de las 5 Variables Mas Correlacionadas con el Target\\n'\n",
        "    '(histograma + KDE por clase)',\n",
        "    fontsize=14, fontweight='bold', y=1.05\n",
        ")\n",
        "plt.show()"
    ]

    updated = False
    for c in data['cells']:
        if c['cell_type'] == 'code':
            # Identify the cell by checking for section 5 pattern
            if 'DISTRIBUCION DE VARIABLES' in "".join(c['source']) or 'DISTRIBUCION DE LAS 5 VARIABLES' in "".join(c['source']):
                c['source'] = new_source
                updated = True
                break
    
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully with new Section 5 code.")
    else:
        print("Failed to find Section 5 code cell.")

if __name__ == "__main__":
    update_section_5('analisis_explotario.ipynb')
