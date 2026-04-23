import os
from backend.RAG import ingest, retrieve


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs_folder = os.path.join(PROJECT_ROOT, "Documents")

#Step 1 — ingest all PDFs
print("=== INGESTING ===")
for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(docs_folder, filename)
        print(f"Ingesting {filename}...")
        ingest(path)

#Step 2 — test retrieval
print("\n=== RETRIEVING ===")
question = "What is the standard deviation?"
results = retrieve(question)

for i, chunk in enumerate(results):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)