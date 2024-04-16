import streamlit as st
from PIL import Image
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

from src import CFG
from src.embeddings import build_hyde_embeddings
from src.query_expansion import build_multiple_queries_expansion_chain
from src.retrieval_qa import (build_retrieval_qa,build_base_retriever,build_rerank_retriever,build_compression_retriever,)
from src.vectordb import build_vectordb, load_faiss, load_chroma
from streamlit_app.pdf_display import get_doc_highlighted, display_pdf
from streamlit_app.utils import load_base_embeddings, load_llm, load_reranker

st.set_page_config(page_title="Retrieval QA", layout="wide")

LLM = load_llm() #build_llm
BASE_EMBEDDINGS = load_base_embeddings() #build_embeddings
RERANKER = load_reranker()

 #vectordb
@st.cache_resource
def load_vectordb():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)
    raise NotImplementedError

#vectordb hyde
@st.cache_resource
def load_vectordb_hyde(): 
    hyde_embeddings = build_hyde_embeddings(LLM, BASE_EMBEDDINGS)

    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(hyde_embeddings)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(hyde_embeddings)
    raise NotImplementedError

#retriever for the documents: Base, Rerank, Contextual compression
def load_retriever(_vectordb, retrieval_mode): 
    if retrieval_mode == "Base":
        return build_base_retriever(_vectordb)
    if retrieval_mode == "Rerank":
        return build_rerank_retriever(_vectordb, RERANKER)
    if retrieval_mode == "Contextual compression":
        return build_compression_retriever(_vectordb, BASE_EMBEDDINGS)
    raise NotImplementedError


def init_sess_state():
    if "uploaded_filename" not in st.session_state:
        st.session_state["uploaded_filename"] = ""

    if "last_form" not in st.session_state:
        st.session_state["last_form"] = list()

    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    if "last_response" not in st.session_state:
        st.session_state["last_response"] = dict()

    if "last_related" not in st.session_state:
        st.session_state["last_related"] = list()


def doc_qa():
    init_sess_state()

    with st.sidebar:
        st.header("RAG with quantized LLM")
        image = Image.open('./assets/francisco.png')
        st.image(image, caption='Don Francisco de Arobe')
        st.info(f"LLM: `{CFG.LLM_PATH}`")
        st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
        st.info(f"Reranker: `{CFG.RERANKER_PATH}`")
        

        uploaded_file = st.file_uploader("Sube un PDF y crea un VectorDB", type=["pdf"]) #Upload a PDF and build VectorDB
        
        if st.button("Construir VectorDB"):
            if uploaded_file is None:
                st.error("No se ha subido un PDF")
            else:
                uploaded_filename = f"./data/{uploaded_file.name}"
                with open(uploaded_filename, "wb") as f:
                    f.write(uploaded_file.getvalue())
                with st.spinner("Construyendo VectorDB..."):
                    build_vectordb(uploaded_filename)
                st.session_state.uploaded_filename = uploaded_filename

        if st.session_state.uploaded_filename != "":
            st.info(f"Documento actual: {st.session_state.uploaded_filename}")

        try:
            with st.status("Cargar VectorDB", expanded=False) as status:
                st.write("Cargando VectorDB ...")
                vectordb = load_vectordb()
                st.write("Cargando HyDE VectorDB ...")
                vectordb_hyde = load_vectordb_hyde()
                status.update(
                    label="Carga completada!", state="complete", expanded=False
                )

            st.success("Leyendo de VectorDB existente")
        except Exception as e:
            st.error(f"No se ha encontrado un VectorDB existente: {e}")

    c0, c1 = st.columns(2)

    with c0.form("qa_form"):
        user_query = st.text_area("Tu pregunta")
        with st.expander("Configuración"):
            mode = st.radio(
                "Modo",
                ["Retrieval only", "Retrieval QA"],
                index=1,
                help="""Retrieval solo extrae los fragmentos importante de los documentos, \
                mientras que Retrieval QA dará una respuesta a tu pregunta y por lo tanto tardará más en ejecutarse.""",
            )
            retrieval_mode = st.radio(
                "Método de Retrieval",
                ["Base", "Rerank", "Compresión contextual"],
                index=1,
            )
            use_hyde = st.checkbox("Usar HyDE (solo compatible con Retrieval QA)")

        submitted = st.form_submit_button("Preguntar")
        if submitted:
            if user_query == "":
                st.error("Por favor introduzca una pregunta.")

    if user_query != "" and (
        st.session_state.last_query != user_query
        or st.session_state.last_form != [mode, retrieval_mode, use_hyde]):
            
        st.session_state.last_query = user_query
        st.session_state.last_form = [mode, retrieval_mode, use_hyde]

        if mode == "Retrieval only":
            retriever = load_retriever(vectordb, retrieval_mode)
            with c0:
                with st.spinner("Retrieving ..."):
                    relevant_docs = retriever.get_relevant_documents(user_query)

            st.session_state.last_response = {
                "query": user_query,
                "source_documents": relevant_docs,
            }

            chain = build_multiple_queries_expansion_chain(LLM)
            res = chain.invoke(user_query)
            st.session_state.last_related = [
                x.strip() for x in res.split("\n") if x.strip()
            ]
        else:
            db = vectordb_hyde if use_hyde else vectordb
            retriever = load_retriever(db, retrieval_mode)
            retrieval_qa = build_retrieval_qa(LLM, retriever)

            st_callback = StreamlitCallbackHandler(
                parent_container=c0.container(),
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
            )
            st.session_state.last_response = retrieval_qa.invoke(
                user_query, config=RunnableConfig(callbacks=[st_callback])
            )
            st_callback._complete_current_thought()

    if st.session_state.last_response:
        with c0:
            st.warning(f"##### {st.session_state.last_query}")
            if st.session_state.last_response.get("result") is not None:
                st.success(st.session_state.last_response["result"])

            if st.session_state.last_related:
                st.write("#### Relacionado")
                for r in st.session_state.last_related:
                    st.write(f"```\n{r}\n```")

        with c1:
            st.write("#### Fuentes")
            for row in st.session_state.last_response["source_documents"]:
                st.write("**Página {}**".format(row.metadata["page"] + 1))
                st.info(row.page_content.replace("$", r"\$"))

            # Display PDF
            st.write("---")
            _display_pdf_from_docs(st.session_state.last_response["source_documents"])


def _display_pdf_from_docs(source_documents):
    n = len(source_documents)
    i = st.radio(
        "Ver en PDF", list(range(n)), format_func=lambda x: f"Extract {x + 1}"
    )
    row = source_documents[i]
    try:
        extracted_doc, page_nums = get_doc_highlighted(
            row.metadata["source"], row.page_content
        )
        if extracted_doc is None:
            st.error("Página no encontrada")
        else:
            display_pdf(extracted_doc, page_nums[0] + 1)
    except Exception as e:
        st.error(e)


if __name__ == "__main__":
    doc_qa()
