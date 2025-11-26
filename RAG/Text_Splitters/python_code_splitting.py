from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = '''
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.7,
) # type: ignore

model = ChatHuggingFace(llm=llm)

response = model.invoke("do you know about ram")
print(response.content)
'''
splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size=100,
    chunk_overlap =0
)
   

result = splitter.split_text(code)

# for splitting text from documnets which we will get from document_loaders, we use the .split_document() 

print(len(result))
print(result)