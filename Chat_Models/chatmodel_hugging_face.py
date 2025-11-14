from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # huggingface endpoint used when we want to use huggingface API.
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    model_kwargs=(
        temperature = 0.1,
        max_new_tokens=100
    )
)
model  = ChatHuggingFace(llm = llm)

response  = model.invoke("what is the capital of bihar?")
print(response.content)

