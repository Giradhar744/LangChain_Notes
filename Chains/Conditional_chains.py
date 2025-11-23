from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model= 'gemini-2.5-flash'
)


parser = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['Positive','Negative'] = Field(description= 'Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object= feedback)

prompt1 = PromptTemplate(
    template= 'Classify the sentiment of the following feedback text into positive and negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables= {'format_instruction':parser2.get_format_instructions()}
)
classifier_chain = prompt1 | model1 | parser2


prompt2  = PromptTemplate(
    template='Write an appropriate response to this positive feedback and only one best response \n {feedback}',
    input_variables=['feedback']
)

prompt3  = PromptTemplate(
    template='Write an appropriate response to this negative feedback and only one best response \n {feedback}',
    input_variables=['feedback']
)

# If else statemnet
branch_chain = RunnableBranch(
    (lambda x:x.sentiment =='Positive',  prompt2| model1 |parser),#  type: ignore  (condition, chain)
    (lambda x:x.sentiment =='Negative',  prompt3| model1 |parser),#  type: ignore   (condition, chain)
    RunnableLambda(lambda x: "Could not find sentiment")
)  

final_chain = classifier_chain| branch_chain

res = final_chain.invoke({'feedback':'This is terrible phone'})

print(res)

final_chain.get_graph().print_ascii()


