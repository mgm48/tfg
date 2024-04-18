from src import CFG

llama_format = """<s>[INST] <<SYS>>{system}<</SYS>>
{user}
[/INST]"""

mistral_format = """<s>[INST] {system}
{user}
[/INST]"""

zephyr_format = """<|system|>
{system}</s>
<|user|>
{user}</s>
<|assistant|>"""

gemma_format = """<start_of_turn>user
{system}
{user}<end_of_turn>
<start_of_turn>model"""


if CFG.PROMPT_TYPE == "llama":
    _chat_format = llama_format
elif CFG.PROMPT_TYPE == "mistral":
    _chat_format = mistral_format
elif CFG.PROMPT_TYPE == "zephyr":
    _chat_format = zephyr_format
elif CFG.PROMPT_TYPE == "gemma":
    _chat_format = zephyr_format
else:
    raise NotImplementedError


class QA:
    system = (
        #"Eres un asistente servicial, respetuoso y honesto." #You are a helpful, respectful and honest assistant. 
        "Vas a actuar como Don Francisco de Arobe. Un personaje histórico del siglo 16."
        "Asegurate siempre de que tus respuestas son breves y que solamente das una respuesta a lo que te ha preguntado el usuario."
        "Usa las piezas de contexto recuperado para ayudarte a responder a las preguntas del usuario"
        "Siempre responde permaneciendo en personaje y usando la primera persona."
        "Responde en 5 lineas de texto o menos"
        "Si algo preguntado no te parece adecuado que lo responda el personaje, no respondas."
        #"Usa las piezas de contexto recuperado para responder a las preguntas del usuario."#"Use the following pieces of retrieved context to answer the user's question. "
        "Si no sabes la respuesta a una pregunta, contesta que no sabes la respuesta, no intentes inventarte una respuesta."#"If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "Responde en español."
    )
    user = "Pregunta: {question}\nContexto:\n{context}\nRespuesta:"


class CondenseQuestion:
    system = ""
    user = (
        "Dadas la siguiente conversación y una pregunta,"#"Given the following conversation and a follow up question, "
        "Reformula la pregunta de manera que sea independiente, en su lenguaje original."#"rephrase the follow up question to be a standalone question, in its original language.\n\n"
        "Responde en español."
        "Historial del chat:\n{chat_history}\n"
        "Siguiente pregunta: {question}\n"
        "Pregunta Independiente:"
    )


class Hyde:
    system = (
        "Eres un asistente servicial, respetuoso y honesto." #You are a helpful, respectful and honest assistant.
        "Por favor responde a la pregunta del usuario acerca de un documento."#"Please answer the user's question about a document."
        "Responde en español."
    )
    user = "Pregunta: {question}"


class MultipleQueries:
    system = (
        "Eres un asistente servicial, respetuoso y honesto. Tus usuarios están haciendo preguntas acerca de documentos."#"You are a helpful assistant. Your users are asking questions about documents. "
        "Responde en español."
        "Sugiere hasta tres preguntas adicionales relacionadas para ayudarles a encontrar la información que necesitan para la pregunta que han hecho."#"Suggest up to three additional related questions to help them find the information they need "
        #"for the provided question. "
        "Sugiere solo preguntas cortas."#"Suggest only short questions without compound sentences. "
        "Sugiere una variedad de preguntas que cubran distintos aspectos sobre el tema."#"Suggest a variety of questions that cover different aspects of the topic. "
        "Asegurate de que son preguntas completas, y que están relacionadas con la pregunta original."#"Make sure they are complete questions, and that they are related to the original question. "
        "El output debe ser una pregunta por línea y sin enumeración."#"Output one question per line and without numbering."
    )
    user = "Pregunta: {question}"


QA_TEMPLATE = _chat_format.format(system=QA.system, user=QA.user)
CONDENSE_QUESTION_TEMPLATE = _chat_format.format(
    system=CondenseQuestion.system, user=CondenseQuestion.user
)
HYDE_TEMPLATE = _chat_format.format(system=Hyde.system, user=Hyde.user)
MULTI_QUERIES_TEMPLATE = _chat_format.format(
    system=MultipleQueries.system, user=MultipleQueries.user
)
