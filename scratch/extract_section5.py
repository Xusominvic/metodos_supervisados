import json

def get_section_5():
    with open('analisis_explotario.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for i, c in enumerate(nb['cells']):
        if '## 5. Distribucion de las Variables' in ''.join(c['source']):
            found = True
        elif found and c['cell_type'] == 'code':
            with open('scratch/section_5.txt', 'w', encoding='utf-8') as out:
                out.write("".join(c['source']))
            return

if __name__ == "__main__":
    get_section_5()
