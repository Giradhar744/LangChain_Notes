from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.5,
)  #type: ignore

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a Good AI Assistant")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input)) #type: ignore

    if user_input == "exit":
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content)) #type: ignore 

    print("AI:", result.content)

print(chat_history)
