from langchain_openai import OpenAI
from dotenv import load_dotenv       # It helps to load the keys from secret files to current file.

load_dotenv()

llm = OpenAI(model= 'gpt-3.5-turbo-instruct')

response  = llm.invoke("What is capital of india")  # by the invoke method, we communicate with the different- different models.

print(response)