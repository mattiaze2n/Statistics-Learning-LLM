from dotenv import load_dotenv
load_dotenv()


from backend.llm_client import ask


# --- Simple single-turn example (no history, no RAG) ---
response = ask(
    user_message="Explain what model selection for logistic regression",
    user_level="Beginner",
    user_goal="pass a university statistics exam",
    user_background="biology"
)

if "error" in response:
    print(f"Error: {response['error']}")
else:
    print(response["answer"])


# --- Example with RAG and conversation history ---
# from backend.rag_pipeline import ingest, retrieve
#
# ingest("data/my_document.pdf")  # run once to populate the vector store
#
# history = []
# while True:
#     user_input = input("You: ").strip()
#     if not user_input:
#         break
#     chunks = retrieve(user_input)
#     result = ask(
#         user_message=user_input,
#         user_level="Intermediate",
#         conversation_history=history,
#         rag_chunks=chunks,
#     )
#     if "error" in result:
#         print(f"Error: {result['error']}")
#     else:
#         print(f"Assistant: {result['answer']}")
#         history = result["updated_history"]