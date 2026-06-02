import nbformat

def find_cell_with_text(nb_path, search_text):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' or cell.cell_type == 'markdown':
            if search_text in cell.source:
                print(f"Found in cell {i}:")
                print(cell.source)
                print("-" * 40)

if __name__ == "__main__":
    find_cell_with_text('analisis_explotario.ipynb', 'high_corr')
