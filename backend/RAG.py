import os
import time
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

    After this runs, ChromaDB holds vectors for every chunk of this document
    on disk in chroma_db/. retrieve() can then load that store without any
    re-embedding.

    Args:
        file_path: Path to a .pdf, .txt, or .md file.

    Returns:
        The number of chunks stored.
    """
    docs = _load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            Chroma.from_documents(
                documents=batch,
                embedding=_embeddings,
                persist_directory=_CHROMA_DIR,
            )
            print(f"  Embedded chunks {i+1} to {i+len(batch)}/{len(chunks)}")
            time.sleep(2)  # 2 second pause between batches
        except Exception as e:
            print(f"Ingestion failed at batch {i}: {e}")
            raise

    print(f"Ingested {len(chunks)} chunks from '{file_path}' into ChromaDB.")
    return len(chunks)
    


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
    
