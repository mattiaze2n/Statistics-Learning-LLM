import os
import time
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

from .config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, GOOGLE_API_KEY


if GOOGLE_API_KEY:
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

_CHROMA_DIR = _CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db"
)

_embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")


def _load_document(file_path: str):
    """
    Load a single file and return a list of LangChain Document objects.

    Each Document has:
      .page_content  — the raw text
      .metadata      — dict with 'source' (path) and 'page' (PDFs only)

    This metadata is preserved on every chunk after splitting, so the app
    can always tell the user which file and page an answer came from.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        loader = PyMuPDFLoader(file_path)
    elif ext in ('.txt', '.md'):
        loader = TextLoader(file_path, encoding = 'utf-8')
    else:
        raise NotImplementedError
    return loader.load()


def ingest(file_path: str) -> int:
    """
    Load → split → embed → persist.
    Uses document IDs to avoid re-embedding already stored chunks.
    """
    filename = os.path.basename(file_path)
    docs = _load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    ids = [f"{filename}::chunk{i}" for i in range(len(chunks))]

    vectorstore = Chroma(
        persist_directory=_CHROMA_DIR,
        embedding_function=_embeddings,
    )

    existing_ids = set(vectorstore.get()["ids"])
    
    new_chunks = []
    new_ids = []
    for chunk, id in zip(chunks, ids):
        if id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(id)

    if not new_chunks:
        print(f"'{filename}' already fully ingested — skipping.")
        return 0

    batch_size = 50
    for i in range(0, len(new_chunks), batch_size):
        batch_chunks = new_chunks[i:i + batch_size]
        batch_ids    = new_ids[i:i + batch_size]
        try:
            vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
            print(f"  Embedded chunks {i+1} to {i+len(batch_chunks)}/{len(new_chunks)}")
            time.sleep(2)
        except Exception as e:
            print(f"Ingestion failed at batch {i}: {e}")
            raise

    print(f"Ingested {len(new_chunks)} new chunks from '{filename}'.")
    return len(new_chunks)
    


def retrieve(query: str, top_k: int = TOP_K_RESULTS) -> list[str]:
    """
    Embed the query → find similar chunks → return their text.

    The list[str] this returns is the exact shape that llm_client.ask()
    expects as its rag_chunks= argument, which prompts.build_rag_context()
    then formats into the context block prepended to the user message.

    Args:
        query: The user's question.
        top_k: How many chunks to return (default from config.py).

    Returns:
        A list of raw text strings — one per retrieved chunk.
    """ 
    top_k_documents = Chroma(
        persist_directory = _CHROMA_DIR,
        embedding_function=_embeddings,
    )
    results = top_k_documents.similarity_search(query, k = top_k)
    return [doc.page_content for doc in results]
    
