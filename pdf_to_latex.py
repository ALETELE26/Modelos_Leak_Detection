import subprocess
import os

print(f"Ejecutando script desde: {os.path.abspath(__file__)}")
def convert_pdf_to_latex(pdf_path, output_dir=None):
    """Convierte PDF a LaTeX usando pdftotext y genera un documento básico."""
    
    # Verificar que el archivo existe
    if not os.path.exists(pdf_path):
        print(f"Error: El archivo '{pdf_path}' no existe")
        print(f"Ruta absoluta: {os.path.abspath(pdf_path)}")
        return None
        
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    tex_path = os.path.join(output_dir, f"{base_name}.tex")
    
    try:
        # Verificar si pdftotext está instalado
        result = subprocess.run(['where', 'pdftotext'], 
                                capture_output=True, text=True)
        if "pdftotext" not in result.stdout:
            print("Error: pdftotext no está instalado o no está en el PATH")
            print("Instala Poppler para Windows: https://github.com/oschwartz10612/poppler-windows/releases/")
            return None
        
        # Extraer texto del PDF
        print(f"Extrayendo texto de: {pdf_path}")
        subprocess.run(['pdftotext', pdf_path, txt_path], check=True)
        
        # Verificar que se creó el archivo de texto
        if not os.path.exists(txt_path):
            print(f"Error: No se pudo crear el archivo de texto: {txt_path}")
            return None
            
        # Leer el texto extraído
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Crear documento LaTeX básico
        latex_content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\title{{{base_name}}}
\\begin{{document}}
\\maketitle

{text_content}

\\end{{document}}
"""
        
        # Guardar el documento LaTeX
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Conversión completada: {tex_path}")
        return tex_path
        
    except Exception as e:
        print(f"Error durante la conversión: {str(e)}")
        return None

# Ubicación correcta del PDF (asegúrate de que esta sea la ruta correcta)
current_dir = os.path.dirname(os.path.abspath(__file__))
input_pdf = r"c:\Users\Cuba\Documents\uni\STM32\fast_cwt\fCWT\Modelos\BoundEffRed.pdf"
# Verifica que existe:
print(f"Verificando archivo: {input_pdf}")
print(f"El archivo existe: {os.path.exists(input_pdf)}")
# Llamar a la función
result = convert_pdf_to_latex(input_pdf)
if result:
    print(f"Archivo LaTeX generado: {result}")
else:
    print("La conversión falló")