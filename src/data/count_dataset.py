import pandas as pd
from pathlib import Path

def count_clean_dataset(base_path):
    # Lista para almacenar los resultados
    data = []
    
    # Validar que el directorio principal exista
    if not base_path.exists():
        print(f"Error: No se encontró la ruta {base_path}")
        return
        
    # Iterar sobre las carpetas de las clases
    for class_dir in sorted(base_path.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Contadores por entorno
            count_lab = 0
            count_real = 0
            
            # Revisar carpeta 'lab'
            lab_dir = class_dir / 'lab'
            if lab_dir.exists():
                count_lab = len([f for f in lab_dir.iterdir() if f.is_file()])
                
            # Revisar carpeta 'real'
            real_dir = class_dir / 'real'
            if real_dir.exists():
                count_real = len([f for f in real_dir.iterdir() if f.is_file()])
            
            # Calcular total
            total_class = count_lab + count_real
            
            # Agregar a la lista de datos
            data.append({
                'Clase': class_name,
                'Lab': count_lab,
                'Real': count_real,
                'Total': total_class
            })
            
    # Crear un DataFrame para mostrarlo ordenado
    df = pd.DataFrame(data)
    
    # Calcular totales globales
    total_lab = df['Lab'].sum()
    total_real = df['Real'].sum()
    total_general = df['Total'].sum()
    
    # Imprimir resultados en formato tabla Markdown
    print("\n### Resultados del Conteo\n")
    print(df.to_markdown(index=False))
    
    print("\n### Resumen Global")
    print(f"* **Total Imágenes Lab:** {total_lab}")
    print(f"* **Total Imágenes Real:** {total_real}")
    print(f"* **Total General:** {total_general}")

if __name__ == "__main__":
    # 1. Obtiene la ruta absoluta de dónde está este script (mi_proyecto/src/data/)
    script_dir = Path(__file__).resolve().parent
    
    # 2. Navega hacia la raíz (sube de 'data' a 'src', y de 'src' a la raíz) y entra a data/clean
    ruta_clean = script_dir.parent.parent / "data" / "clean"
    
    print(f"Buscando imágenes en: {ruta_clean}")
    
    # Ejecutar el conteo
    count_clean_dataset(ruta_clean)