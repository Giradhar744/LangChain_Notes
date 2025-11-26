from langchain_community. document_loaders import WebBaseLoader
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
    template= 'Answer the Following Question\n{question} from the following {text}',
    input_variables= ['question','text']
)

parser = StrOutputParser()


url = 'https://www.a mazon.in/Apple-MacBook-13-inch-10-core-Unified/dp/B0DZDDQ429/ref=sr_1_1_sspa?crid=3GZZ0XW09C1K5&dib=eyJ2IjoiMSJ9.cxg64j71asHtIpoHuVSkM44XuP9BEolZupPa0pi5dGOAlCNAHlQCwrBDLwsm8n1AnbX5JCctr3aygtETgg1IBYJIKV5TCVqJn77MR_Zc9OfdmvGk_LfHOFlthSt_JRPIBxGQJI7ICk1Qxq7GPWztkwXIsUDzb2KflAb11VUCGruMFXJQIevX7M34bZHo7tbSQ1n_6q2SlVW_IHH9Al0BjnmY_g06oQHlMPn0k5jd0_c._I5NVI38H7LKxMqWM-ZM1ghjEpsacApEZvBKHtJqtNE&dib_tag=se&keywords=macbook%2Bair%2Bm4&qid=1764161717&sprefix=macbook%2Caps%2C295&sr=8-1-spons&aref=7XLVAqVBAB&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1'
loader = WebBaseLoader(url)
docs  = loader.load()
data = docs[0].page_content

chain = prompt| model| parser

result = chain.invoke({'question':'what is the specification of this laptop','text':data})

print(result)