from dotenv import load_dotenv
load_dotenv()


from backend.llm_client import ask
from backend.RAG import retrieve 

user_message = 'Explain standard deviation'
chunks = retrieve(user_message)

response = ask(
    user_message= user_message,
    user_level="Beginner",
    user_goal="pass a university statistics exam",
    user_background="biology",
    rag_chunks = chunks
)

if "error" in response:
    print(f"Error: {response['error']}")
else:
    print(response["answer"])


