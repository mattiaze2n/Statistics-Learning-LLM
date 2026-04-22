from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .config import GEMINI_MODEL, TEMPERATURE, MAX_TOKENS
from .prompts import build_system_prompt, build_user_prompt, build_rag_context

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=TEMPERATURE,
    max_output_tokens=MAX_TOKENS,
    streaming = True
)

# Prompt template — system prompt + optional user level + full chat history + user message
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_message}"),
])

chain = prompt_template | llm | StrOutputParser()


def _convert_history(conversation_history: list[dict]) -> list:
    """Convert a list of {'role': ..., 'content': ...} dicts to LangChain message objects."""
    lc_messages = []
    for msg in conversation_history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] in ("assistant", "model"):
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


def ask(
    user_message: str,    
    user_level: str = "Intermediate", 
    user_goal: str = "learn data science",
    user_background: str = "not specified",
    conversation_history: list[dict] | None = None,
    rag_chunks: list[str] | None = None
) -> dict:
    """
    Send a message to the LLM and return the answer plus updated history.
 
    Args:
        user_message:         The user's raw question.
        user_level:           'Beginner', 'Intermediate', or 'Advanced'
        user_goal:            e.g. 'pass a university exam'
        user_background:      e.g. 'biology', 'economics'
        conversation_history: Previous turns as list of role/content dicts.
        rag_chunks:           Retrieved context chunks (optional).
 
    Returns:
        {'answer': str, 'updated_history': list} on success
        {'error': str} on failure
    """

    if conversation_history is None:
        conversation_history = []

    #Build Personalized system prompt 
    system_prompt = build_system_prompt(
        user_level = user_level,
        user_goal = user_goal,
        user_background = user_background,
    )

    rag_context = build_rag_context(rag_chunks or [])

    composed_user_message = build_user_prompt(
        user_question = rag_context + user_message,
        user_level = user_level,
        user_goal = user_goal,
        user_background = user_background
    )
    
    lc_history = _convert_history(conversation_history)

    try:
        answer = chain.invoke({
            "system_prompt": system_prompt,
            "chat_history": lc_history,
            "user_message": composed_user_message,
        })

        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]

        return {"answer": answer, "updated_history": updated_history}

    except Exception as e:
        print(f"LLM API error: {e}")
        return {"error": str(e)}
    
def ask_stream(
        user_message: str,
        user_level : str = 'Intermediate',
        user_goal: str = 'learn data science',
        user_background : str = 'not specified',
        conversation_history: list[dict] | None = None, 
        rag_chunks : list[str] | None = None,
    ):

    """
    Generator version of ask(). 
    Yields text chunks as they arrive from the LLM.
    """

    if conversation_history is None: 
        conversation_history = []
    
    system_prompt = build_system_prompt(
        user_level = user_level,
        user_goal = user_goal,
        user_background = user_background,
    )

    rag_context = build_rag_context(rag_chunks or [])

    composed_user_message = build_user_prompt(
        user_question = rag_context + user_message,
        user_level = user_level,
        user_goal = user_goal,
        user_background = user_background
    )

    messages = [SystemMessage(content = system_prompt)]
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content = msg["content"]))
        else:
            messages.append(AIMessage(content = msg ["content"]))
    messages.append(HumanMessage(content = composed_user_message))

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content

    

