"""
app.py Streamlit for Statistics tutor
"""

import streamlit as st
from dotenv import load_dotenv
import json 
import os 


load_dotenv()

from backend.llm_client import ask_stream
from backend.RAG import ingest, retrieve
from backend.eval_tab import render_eval_tab

INGESTED_FILES_RECORD = "ingested_files.json"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataTutor",
    page_icon="📊",
    layout="wide",
)


PREFS_FILE  = "user_prefs.json"

DEFAULT_PREFERENCES = {
    "user_level":      "Intermediate",
    "user_goal":       "learn data science",
    "user_background": "not specified",
}

def estimate_tokens(text:str) -> int:
    return len(text) // 4

def load_preferences() -> dict:
    """
    Read Preferences from disk
    """

    if not os.path.exists(PREFS_FILE):
        return DEFAULT_PREFERENCES
    
    try:
        with open(PREFS_FILE, "r", encoding = "utf-8") as f:
            saved = json.load(f)

        return{**DEFAULT_PREFERENCES, **saved}
    except (json.JSONDecodeError, IOError):
        return DEFAULT_PREFERENCES.copy()

def save_preferences():
    """
    Write current profile values from session state 
    to disk. This is called everytime a profile field changes
    """

    prefs = {
        "user_level": st.session_state.user_level,
        "user_goal": st.session_state.user_goal,
        "user_background": st.session_state.user_background,
    }

    with open(PREFS_FILE, "w", encoding = "utf-8") as f:
        json.dump(prefs, f, indent = 2)

def load_ingested_files() -> list[str]:
    """
    Read the list of previously ingested filenames from disk.
    Returns an empty list if files do not exist.
    """
    if not os.path.exists(INGESTED_FILES_RECORD):
        return []
    try:
        with open(INGESTED_FILES_RECORD, "r", encoding = "utf-8") as f:
            data = json.load(f)
        return data.get("files", [])
    except (json.JSONDecodeError, IOError):
        return []

def add_ingested_file(filename: str):
    """
    Append a filename to the record.
    Avoids uplicates.
    """
    current = load_ingested_files()
    if filename not in current:
        current.append(filename)
        with open(INGESTED_FILES_RECORD, "w", encoding = "utf-8") as f:
            json.dump({"files": current}, f, indent = 2)

def init_session_state():
    prefs = load_preferences()

    defaults = {
        "conversation_history": [],   
        "user_level":           prefs["user_level"],
        "user_goal":            prefs["user_goal"],
        "user_background":      prefs["user_background"],
        "use_rag":              False,
        "ingested_files":       load_ingested_files(),  
        "total_input_tokens":   0,    
        "total_output_tokens":  0,
        "is_thinking":          False,  
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# 2. SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.title("⚙️  Your Profile")

        st.selectbox(
            "Your level",
            options=["Beginner", "Intermediate", "Advanced"],
            index=["Beginner", "Intermediate", "Advanced"].index(
                st.session_state.user_level
            ),
            key = "user_level",
            on_change = save_preferences,
            help="Controls vocabulary, depth, and example complexity.",
        )

        st.text_input(
            "Your learning goal",
            value=st.session_state.user_goal,
            placeholder="e.g. pass a university exam",
            key = "user_goal",
            on_change = save_preferences,
        )

        st.text_input(
            "Your background",
            value=st.session_state.user_background,
            key = "user_background",
            on_change = save_preferences,
            placeholder="e.g. biology, economics",
        )

        st.divider()

        # ── Documents + RAG ───────────────────────────────────────────────────
        st.subheader("📄 Documents")

        st.session_state.use_rag = st.toggle(
            "Answer from my documents",
            value=st.session_state.use_rag,
        )

        uploaded_file = st.file_uploader(
            "Upload PDF, TXT, or MD",
            type=["pdf", "txt", "md"],
        )

        if uploaded_file is not None:
            _handle_file_upload(uploaded_file)

        if st.session_state.ingested_files:
            st.caption("Ingested this session:")
            for fname in st.session_state.ingested_files:
                st.markdown(f"- ✅ `{fname}`")

        st.divider()

        # ── Token / cost display ──────────────────────────────────────────────
        st.subheader("📈 Usage this session")
        st.metric("Input tokens",  st.session_state.total_input_tokens)
        st.metric("Output tokens", st.session_state.total_output_tokens)

        st.divider()

        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.total_input_tokens  = 0
            st.session_state.total_output_tokens = 0
            st.rerun()


def _handle_file_upload(uploaded_file):
    """Save uploaded file to a temp path, ingest it, track the name."""
    import tempfile, os

    if uploaded_file.name in st.session_state.ingested_files:
        st.sidebar.info(f"`{uploaded_file.name}` already ingested.")
        return

    with st.sidebar.status(f"Ingesting `{uploaded_file.name}`…"):
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            n_chunks = ingest(tmp_path)
            st.session_state.ingested_files.append(uploaded_file.name)
            add_ingested_file(uploaded_file.name)
            st.sidebar.success(f"Done — {n_chunks} chunks stored.")
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")
        finally:
            os.unlink(tmp_path)


# =============================================================================
# 3. CHAT VIEW
# =============================================================================

def render_chat():
    st.title("📊 DataTutor")
    st.caption(
        f"Level: **{st.session_state.user_level}** · "
        f"Goal: *{st.session_state.user_goal}* · "
        f"Background: *{st.session_state.user_background}*"
    )

    # ── Render history ────────────────────────────────────────────────────────
    for message in st.session_state.conversation_history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a data science question…")

    if user_input and not st.session_state.is_thinking:
        _handle_user_message(user_input)


def _handle_user_message(user_input: str):
    """
    Full request-response cycle for one user turn:
      1. Show user bubble.
      2. Retrieve RAG chunks (optional).
      3. Call LLM backend.
      4. Show response — plain text today, streaming tomorrow (TODO-2).
      5. Update session state.
    """
    st.session_state.is_thinking = True

    # 1. Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. RAG retrieval
    rag_chunks = []
    if st.session_state.use_rag:
        with st.spinner("Searching your documents…"):
            try:
                rag_chunks = retrieve(user_input)
            except Exception as e:
                st.warning(f"RAG retrieval failed — answering without documents. ({e})")

        if rag_chunks:
            with st.expander(f"Retrieved {len(rag_chunks)} chunks from your document"):
                for i, chunk in enumerate(rag_chunks):
                    st.markdown(f"**Chunk {i+1}**")
                    st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                    if i < len(rag_chunks) -1:
                        st.divider()

    # 3 & 4. LLM call + display
    with st.chat_message("assistant"):
        answer = st.write_stream(
            ask_stream(           
                        user_message=user_input,
                        user_level=st.session_state.user_level,
                        user_goal=st.session_state.user_goal,
                        user_background=st.session_state.user_background,
                        conversation_history=st.session_state.conversation_history,
                        rag_chunks=rag_chunks or None,
            )
        )

    # 5. Update history
    st.session_state.conversation_history = st.session_state.conversation_history + [
        {"role" : "user", "content" : user_input},
        {"role" : "assistant", "content": answer},
    ]

    st.session_state.total_input_tokens += estimate_tokens(user_input)
    st.session_state.total_output_tokens += estimate_tokens(answer)
    st.session_state.is_thinking = False

# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    init_session_state()
    render_sidebar()

    tab_chat, tab_eval = st.tabs(["Chat", "Eval"])

    with tab_chat:
        render_chat()
    with tab_eval:
        render_eval_tab()


if __name__ == "__main__":
    main()