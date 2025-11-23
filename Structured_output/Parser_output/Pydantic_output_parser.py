from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) # type:ignore

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description='Name of the Person')
    age: int = Field(gt=18, description='Age of the Person')
    city: str= Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(
   template=(
        "Generate ONLY valid JSON.\n"
        "Do NOT add any text before or after.\n"
        "Generate the name, age, and city of a fictional {place} person.\n"
        "{format_instruction}"
    ),
    input_variables= ['place'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}

)

# prompt = template.invoke({'place':'India'})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content) # type: ignore
# print(final_result)

chain = template|model | parser

result = chain.invoke({'place':'Indian'})
print(result)