import os

import streamlit as st
import speech_recognition as sr
import pyttsx3
from PIL import Image

from src import CFG
from src.retrieval_qa import build_retrieval_chain
from src.vectordb import build_vectordb, delete_vectordb, load_faiss, load_chroma
from streamlit_app.utils import perform, load_base_embeddings, load_llm, load_reranker

from src.audio_player import AudioManager

st.set_page_config(page_title="Conversación con Don Francisco de Arobe",layout="wide")
dev = CFG.DEV_MODE
LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()

@st.cache_resource
def load_vectordb():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)
    raise NotImplementedError

#Engines para tts y stt
audio_manager = AudioManager()
engine = pyttsx3.init()
r = sr.Recognizer()

#Containers de steamlit
c = st.container(height=410,border=False)
c_extra = st.container(height=60,border=False)
ee = c_extra.empty()

if 'texto' not in st.session_state:
    st.session_state['texto'] = ""
#if "uploaded_filename" not in st.session_state:
    #st.session_state["uploaded_filename"] = ""

def init_chat_history():
    #Inicializa el historial del chat
    clear_button = st.sidebar.button("Borrar Conversation", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["display_history"] = [("", "Buenos días", None)]

def print_docs(source_documents):
    for row in source_documents:
        st.write(f"**Page {row.metadata['page_number']}**")
        st.info(row.page_content)

def config_tts(v=0, rate=200,vol=0.7,show_voices_info=False):
    voices = engine.getProperty('voices')
    engine.setProperty('rate', rate)     # setting up new voice rate
    engine.setProperty('volume',vol)   # setting up volume   
    engine.setProperty('voice', voices[v].id) # setting voice 
    
    if show_voices_info:
        i = 0
        for voice in voices:
            print("Number:",i)
            print("Voice:",voice.name)
            print(" - ID:",voice.id)
            print(" - Languages:",voice.languages)
            print(" - Gender:",voice.gender)
            print(" - Age:",voice.age)
            print("\n")
            i+=1

def grabar_callback():
    with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            with ee:
                with st.spinner('Escuchando lo que dice...'):
                    audio = r.listen(source)
                st.success('Hecho!')  
    try:
        res = r.recognize_google(audio, language='es-ES')
    except sr.UnknownValueError:
        ee.error("No se ha podido reconocer lo que se ha dicho")
    except sr.RequestError as e:
        ee.error("Ha habido un error con el servicio de reconocimiento de voz; {0}".format(e))
    else:    
        st.session_state['texto'] = res

def borrar_callback():
    st.session_state['texto'] = ""


def doc_conv_qa():
    with st.sidebar:
        st.title("TFG: Conversación IA")
        
        image = Image.open('./assets/francisco.png')
        st.image(image, caption='Don Francisco de Arobe')
        
        with st.expander("Modelos Utilizados"):
            st.info(f"LLM: `{CFG.LLM_PATH}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
            st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        with st.expander("Configuración Respuesta"):
            tts = st.radio(
            "Modo de respuesta",
            ["texto", "texto + voz"],
            index=0,
            captions=["Solo responde usando texto.","Responde tanto con texto como audio."]
            )

        if dev == "dev":
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
        if dev == "user":
            uploaded_file = "./data/franwiki.pdf"
        
        if not os.path.exists(CFG.VECTORDB_PATH):
            st.info("Se debe construir el VectorDB primero.")
            st.stop()

        try:
            with st.status("Carga de datos", expanded=False) as status:
                st.write("Cargando VectorDB...")
                vectordb = load_vectordb()
                st.write("Cargando Retrieval Chain...")
                retrieval_chain = build_retrieval_chain(vectordb, RERANKER, LLM)
                status.update(
                    label="Carga Completada!", state="complete", expanded=False
                )

        except Exception:
            st.error("No se ha encontrado un VectorDB existente")
            st.stop()


    st.sidebar.write("---")
    init_chat_history()
    config_tts()

    # Desplegar historial del chat en container c
    for question, answer, source_documents in st.session_state.display_history:
        if question != "":
            with c:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    st.markdown(answer)

            #if source_documents is not None:
                #with st.expander("Sources"):
                    #print_docs(source_documents)

    c1,c2 = st.columns([9,1])
    with c2:
        grabar = st.button("Grabar",help="Graba con un micrófono lo que quieras preguntarle a la IA",on_click=grabar_callback)
        borrar = st.button("Borrar",help="Borra el contenido de la ventana de texto.",on_click=borrar_callback)

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
                        ee.error("Por favor introduzca una pregunta.") 


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
                    
                st.success('Hecho!') 
                
            ee.empty()    
            with st.chat_message("assistant"):    
                st.markdown(response["answer"])

        st.session_state.chat_history.append((response["question"], response["answer"]))
        st.session_state.display_history.append((response["question"], response["answer"], response["source_documents"]))

        if tts == "texto + voz":
            filepath = "./src/tmp/res.wav"
            engine.save_to_file(response["answer"],filepath)
            engine.runAndWait()
            audio_manager.play_audio(filepath, delete_file=False)

        #Pestaña que te enseña de donde ha salido la respuesta
        #with st.expander("Sources"):
            #print_docs(response["source_documents"])



if __name__ == "__main__":
    doc_conv_qa()
