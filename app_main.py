import streamlit as st
import speech_recognition as sr
import pyttsx3
from PIL import Image
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

from src import CFG
from src.embeddings import build_base_embeddings
from src.llms import build_llm
from src.reranker import build_reranker
from src.retrieval_qa import build_retrieval_chain
from src.vectordb import build_vectordb, load_faiss, load_chroma
from streamlit_app.utils import perform
from src.audio_player import AudioManager

st.set_page_config(page_title="Conversación con Don Francisco de Arobe",layout="wide")

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
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = ""


def init_chat_history():
    #Inicializa el historial del chat
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["source_documents"] = list()


@st.cache_resource
def load_retrieval_chain():
    llm = build_llm()
    embeddings = build_base_embeddings()
    reranker = build_reranker()
    if CFG.VECTORDB_TYPE == "faiss":
        vectordb = load_faiss(embeddings)
    elif CFG.VECTORDB_TYPE == "chroma":
        vectordb = load_chroma(embeddings)
    return build_retrieval_chain(vectordb, reranker, llm)

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
        ee.error("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        ee.error("Could not request results from Google Speech Recognition service; {0}".format(e))
    else:    
        st.session_state['texto'] = res

def borrar_callback():
    st.session_state['texto'] = ""


def doc_conv_qa():
    with st.sidebar:
        st.title("Conversational RAG with quantized LLM")
        image = Image.open('./assets/francisco.png')
        st.image(image, caption='Don Francisco de Arobe')
        st.info(
            f"Usa `{CFG.RERANKER_PATH}` reranker y `{CFG.LLM_PATH}` LLM."
        )
        with st.expander("Configuración"):
            mode = st.radio(
            "Modo de uso",
            ["text", "text + tts"],
            index=0,
            captions=["Solo responde usando texto.","Responde tanto con texto como audio."]
            )

        uploaded_file = st.file_uploader(
            "Sube un PDF para crear un VectorDB", type=["pdf"]
        )
        if st.button("Construir VectorDB"):
            if uploaded_file is None:
                st.error("No hay PDF subido")
            else:
                with st.spinner("Construyendo VectorDB..."):
                    perform(build_vectordb, uploaded_file.read())
                st.session_state.uploaded_filename = uploaded_file.name

        if st.session_state.uploaded_filename != "":
            st.info(f"Documento actual: {st.session_state.uploaded_filename}")

        try:
            with st.status("Carga retrieval_chain", expanded=False) as status:
                st.write("Cargando retrieval_chain...")
                retrieval_chain = load_retrieval_chain()
                status.update(
                    label="Carga Completada!", state="complete", expanded=False
                )

            st.success("Leyendo VectorDB existente")
        except Exception:
            st.error("VectorDB existente no encontrado")


    init_chat_history()
    st.sidebar.write("---")
    config_tts()

    # Desplegar historial del chat en container c
    for (question, answer), source_documents in zip(
        st.session_state.chat_history, st.session_state.source_documents):
        if question != "":
            with c:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    st.markdown(answer)

            #with st.expander("Sources"):
                #for row in source_documents:
                    #st.write("**Page {}**".format(row.metadata["page"] + 1))
                    #st.info(row.page_content)

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
                    response = retrieval_chain.invoke({
                        "question": user_query,
                        "chat_history": st.session_state.chat_history,},
                    )
                st.success('Hecho!') 
                
            ee.empty()    
            with st.chat_message("assistant"):    
                st.markdown(response["answer"])

        st.session_state.chat_history.append((response["question"], response["answer"]))
        st.session_state.source_documents.append(response["source_documents"])

        if mode == "text + tts":
            filepath = "./src/tmp/res.wav"
            engine.save_to_file(response["answer"],filepath)
            engine.runAndWait()
            audio_manager.play_audio(filepath, delete_file=False)

            #Pestaña que te enseña de donde ha salido la respuesta
            #with st.expander("Sources"):
                #for row in response["source_documents"]:
                    #st.write("**Page {}**".format(row.metadata["page"] + 1))
                    #st.info(row.page_content)



if __name__ == "__main__":
    doc_conv_qa()
