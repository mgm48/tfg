# TFG Marta Gago Macías

This is the repository of the app I presented as my Bachelor of Science Thesis in Computer Science. It's an app developed in python that with the use of different AI models creates a chatbot where an user can maintain a conversation with a historical figure about their life and experiences.
The example case used is Francisco de Arobe, an important latin american ruler from the XVI century.

### INSTALLATION INSTRUCTIONS:

#### CONDA ENVIRONMENT:

First you need to install conda: (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Command to create the environment:
```bash
conda env create --name tfg -f environment.yaml --force
```
Command to update the envirnment in case any libraries have been added:
```bash
conda env update --name tfg --file environment.yaml --prune
```
Command to activate and deactivate the environment:
```bash
conda activate tfg
conda deactivate tfg
```

#### AI MODELS

Quantized models can be downloaded from [TheBloke](https://huggingface.co/TheBloke).

### INSTRUCTIONS TO RUN THE APP:

You have to execute the following commands in the anaconda prompt program.
dir represents the local directory where the repository is located.
Example: cd C:\hlocal\tfg
```bash
cd dir
conda activate tfg
streamlit run app.py
```



