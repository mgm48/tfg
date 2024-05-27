import os

import streamlit as st
import speech_recognition as sr
from PIL import Image

from src import CFG
from src.retrieval_qa import build_retrieval_chain
from src.vectordb import build_vectordb, delete_vectordb, load_faiss, load_chroma
from streamlit_app.utils import perform, load_base_embeddings, load_llm, load_reranker

from src.audio_player import AudioManager

st.set_page_config(page_title="Conversación con Don Francisco de Arobe",layout="wide")
user_mode = CFG.DEV_MODE

LLM = load_llm()
RERANKER = load_reranker()
BASE_EMBEDDINGS = load_base_embeddings()

@st.cache_resource
def load_vectordb():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)
    raise NotImplementedError

#Engines para tts y stt
audio_manager = AudioManager()
r = sr.Recognizer()

#Containers de steamlit
c = st.container(height=410,border=False)
c_extra = st.container(height=60,border=False)
ee = c_extra.empty()

if 'texto' not in st.session_state:
    st.session_state['texto'] = ""

def clear_callback():
    st.session_state["backup"]  = st.session_state["chat_history"]
    st.session_state["chat_history"] = list()
    
def restaut_callback():
    if st.session_state["backup"]:
        if not st.session_state["chat_history"]:
            st.session_state["chat_history"] = st.session_state["backup"] 
        else:
            st.session_state["chat_history"] = st.session_state["backup"]  + st.session_state["chat_history"]
        st.session_state["backup"] = list()

def init_chat_history():
    #Inicializa el historial del chat
    clear_button = st.sidebar.button("Borrar Conversación", key="clear",
        help="Borra el historial del chat",on_click=clear_callback)
    restaut_button = st.sidebar.button("Restaurar Conversación", key="restaut",
        help="Restaura el ultimo historial borrado, añadiendo las nuevas consultas que se haya hecho desde entonces.",on_click=restaut_callback)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["display_history"] = [("", "", None)]
    if "backup" not in st.session_state:
        st.session_state["backup"] = list()



def print_docs(source_documents):
    for row in source_documents:
        st.write(f"**Page {row.metadata['page_number']}**")
        st.info(row.page_content)


def grabar_callback():
    with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            with ee:
                with st.spinner('Escuchando lo que dice...'):
                    audio = r.listen(source=source,phrase_time_limit=1000)
                st.success('Grabacción completada')  
    try:
        res = r.recognize_google(audio, language='es-ES')
    except sr.UnknownValueError:
        ee.error("No se ha podido reconocer ningún mensaje.")
        st.session_state['texto'] = ""
    except sr.RequestError as e:
        ee.error("Ha habido un error con el servicio de reconocimiento de voz; {0}".format(e))
        st.session_state['texto'] = ""
    else:
        ee.success("Se ha reconocido el siguiente mensaje. Puede editar su mensaje usando el teclado.")    
        st.session_state['texto'] = res
    ee.empty()

def borrar_texto_callback():
    st.session_state['texto'] = ""

def dev_mode():
    with st.sidebar:
        with st.expander("Modelos Utilizados"):
                st.info(f"LLM: `{CFG.LLM_PATH}`")
                st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
                st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        uploaded_file = st.file_uploader("Sube un PDF para crear un VectorDB", type=["pdf"])
        if st.button("Construir VectorDB"):
            if uploaded_file is None:
                st.error("No hay PDF subido")
                st.stop()

            if os.path.exists(CFG.VECTORDB_PATH):
                st.warning("Borrando VectorDB existente")
                delete_vectordb(CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)
            
            with st.spinner("Construyendo VectorDB..."):
                perform(
                    build_vectordb,
                    uploaded_file.read(),
                    embedding_function=BASE_EMBEDDINGS,
                )
                load_vectordb.clear()

            if not os.path.exists(CFG.VECTORDB_PATH):
                st.info("Se debe construir el VectorDB primero.")
                st.stop()

        


def doc_conv_qa():
    with st.sidebar:
        st.title("Conversación con Don Francisco de Arobe")
        
        image = Image.open('./assets/francisco.png')
        st.image(image, caption='Don Francisco de Arobe')
        
            
        with st.expander("Configuración Sintesis de Voz"):
            tts = st.radio(
            "Modo de respuesta",
            ["texto", "texto + voz"],
            index=0,
            captions=["Solo responde usando texto.","Responde tanto con texto como audio."]
            )


        if user_mode == "dev":
            dev_mode()
        if user_mode == "user":
            uploaded_file = "./data/FranciscodeArobe.pdf"
            if not os.path.exists(CFG.VECTORDB_PATH):
                st.info("Se debe construir el VectorDB primero.")
                perform(
                        build_vectordb,
                        uploaded_file.read(),
                        embedding_function=BASE_EMBEDDINGS,
                    )
                load_vectordb.clear()

        try:
            with st.status("Carga de datos", expanded=False) as status:
                vectordb = load_vectordb()
                st.write("VectorDB: Carga Completada.")
                retrieval_chain = build_retrieval_chain(vectordb, RERANKER, LLM)
                st.write("Cadena de Recuperación: Carga Completada.")
                status.update(
                    label="Sistema de IA: Carga Completada.", expanded=False
                )
        except Exception:
            st.error("La carga del Sistema de IA ha encontrado un error.")
            st.stop()


    st.sidebar.write("---")
    init_chat_history()
    audio_manager.config_tts()
    ee.empty()

    # Desplegar historial del chat en container c
    for question, answer in st.session_state.chat_history:
        if question != "" and answer != "":
            with c:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    st.markdown(answer)

    c1,c2 = st.columns([9,1])
    with c2:
        grabar = st.button("Grabar",key="grabar",help="Graba con un micrófono lo que quieras preguntarle a la IA.\nLa grabación parará de manera automática cuando deje de hablar.",on_click=grabar_callback)
        borrar = st.button("Borrar",key="borrargrab",help="Borra el contenido de la ventana de texto.",on_click=borrar_texto_callback)

    with c1:
        input = st.form("form",clear_on_submit=False,border=True)
        with input:   
            i1,i2 = st.columns([0.85,0.15])  
            with i1:
                user_query = st.text_area("preg",f"",max_chars=1000, key = 'texto', label_visibility="collapsed")

            with i2:
                submitted = st.form_submit_button("Pregunta", use_container_width=True)
                if submitted:
                    if user_query == "":
                        ee.error("Por favor, introduzca una consulta .") 


    if user_query != "" and submitted:
        with c:
            with st.chat_message("user"):
                st.markdown(user_query)
            with ee:
                with st.spinner('Obteniendo respuesta de la IA...'):
                    response = {"question": user_query, "answer": "Buenos días, mi nombre es Francisco de Arobe, un placer.","source_documents": "-"}
                    #response = retrieval_chain.invoke({
                        #"question": user_query,
                        #"chat_history": st.session_state.chat_history,},)
                    
                st.success('Respuesta obtenida.') 
                
            ee.empty()    
            with st.chat_message("assistant"):    
                st.markdown(response["answer"])

        st.session_state.chat_history.append((response["question"], response["answer"]))

        if tts == "texto + voz":
            if not os.path.exists(CFG.TTS_PATH):
                os.mkdir(CFG.TTS_PATH)
            audio_manager.play_tts(CFG.TTS_PATH,response["answer"])




if __name__ == "__main__":
    doc_conv_qa()
