import nbformat

def update_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    target_text = "display(high_corr.reset_index(drop=True))"
    replacement_text = "with pd.option_context('display.max_rows', None):\n        display(high_corr.reset_index(drop=True))"
    
    updated = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and target_text in cell.source:
            # Check if it's already updated
            if "pd.option_context" in cell.source:
                continue
            cell.source = cell.source.replace(target_text, replacement_text)
            updated = True
            print(f"Updated cell content.")
    
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Notebook updated successfully.")
    else:
        print("Target text not found or already updated.")

if __name__ == "__main__":
    update_notebook('analisis_explotario.ipynb')
