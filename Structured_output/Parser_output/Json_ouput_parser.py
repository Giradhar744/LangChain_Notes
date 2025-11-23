from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) # type:ignore

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()
template = PromptTemplate(
    template= 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables= [],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

# prompt  = template.format()  # no input variable (static prompt)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content) # type: ignore

# print(final_result)

# print(type(final_result))


chain = template| model| parser
result1 = chain.invoke({})
print(result1)