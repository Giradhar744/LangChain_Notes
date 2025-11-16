from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} Expert'),
    ('human','Explain about {topic} in 10 points.')

])

# fill theplace holders
prompt = chat_template.invoke({'domain':'cricket', 'topic':'runout'})

print (prompt)
