import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'analisis_explotario.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
cells = nb['cells']
for i, c in enumerate(cells):
    ct = c['cell_type']
    src = ''.join(c['source'])
    # truncate long cells
    if len(src) > 800:
        src = src[:800] + '\n... [TRUNCATED]'
    print(f"--- CELL {i} ({ct}) ---")
    print(src)
    print()
