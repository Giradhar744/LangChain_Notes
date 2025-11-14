from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model =  ChatOpenAI(model  ='gpt-4', temperature=1.5, max_completion_tokens= 50)

response = model.invoke("what is the capital of india?")

# print(response)  # It  returns content means response with the lots of information (metadata)
print(response.content)  # To see only content of the response 

