from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) # type: ignore


model =  ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template= 'write the summary of the  given text{text}\n',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader(file_path= 'C:/Users/girdh/Desktop/Langchain/RAG/Document_Loaders/liscence.txt', encoding= 'utf-8')

docs  = loader.load()
# print(type(docs))
# print(docs[0])
# print(type(docs[0]))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt|model| parser

result  = chain.invoke({'text': docs[0].page_content})
print(result)