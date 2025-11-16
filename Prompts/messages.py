from langchain.messages import AIMessage, HumanMessage, SystemMessage   # these are used to maintain a track of the conversation in chat history, 
                                                                        # means it add label to the messages, so that we can understand that which message is written by human, AI and system message.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.5,)  # type: ignore

model = ChatHuggingFace(llm = llm)

message = [
    SystemMessage(content='You are a good Assistant'),
    HumanMessage(content='Tell me about India')
]

result = model.invoke(message)

message.append(AIMessage(content=result.content))

print(message)

