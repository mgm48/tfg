# TFG Marta Gago Macías
Debe ejecutar los siguientes comandos en el programa anaconda prompt.
dir representa el directorio local donde se ha extraido el repositorio seguido por el nombre del mismo
ejemplo: cd C:\hlocal\tfg
```bash
cd dir
conda activate tfg
streamlit run app.py
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

### MODELOS IA

Quantized models can be downloaded from [TheBloke](https://huggingface.co/TheBloke).





