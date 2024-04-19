# TFG Marta Gago Macías
```bash
cd C:\hlocal\tfg
conda activate tfg
streamlit run app_name.py
```

```bash
conda env create --name rag python=3.11
conda activate rag
pip install -r requirements.txt
```

## INSTRUCCIONES PARA EJECUCIÓN:

### CONDA ENVIRONMENT:

Primero necesitas instalar conda: (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Crear el environment:
```bash
conda env create --name tfg -f environment.yaml --force
```
Actualizar en caso de que se añadan librerías:
```bash
conda env update --name tfg --file environment.yaml --prune
```

Activar y desactivar el environment:
```bash
conda activate tfg
conda deactivate tfg
```





