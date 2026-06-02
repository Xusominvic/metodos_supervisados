import json

def update_notebook_json(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    target_line = '    "    display(high_corr.reset_index(drop=True))\\n",\n'
    # The source is usually a list of strings
    
    updated = False
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            for line in source:
                if 'display(high_corr.reset_index(drop=True))' in line and 'pd.option_context' not in line:
                    indent = line[:line.find('display')]
                    new_source.append(f"{indent}with pd.option_context('display.max_rows', None):\n")
                    new_source.append(f"{indent}    {line.lstrip()}")
                    updated = True
                else:
                    new_source.append(line)
            cell['source'] = new_source
    
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print("Target text not found or already updated.")

if __name__ == "__main__":
    update_notebook_json('analisis_explotario.ipynb')
