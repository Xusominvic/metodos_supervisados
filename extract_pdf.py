import sys
try:
    import pypdf
    reader = pypdf.PdfReader(r"c:\Users\Josep\Documents\metodos_supervisados\doc\enunciado.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except ImportError:
    print("pypdf not installed")
except Exception as e:
    print(f"Error: {e}")
