# Import ChatPromptTemplate for building chat prompts
# Import MessagesPlaceholder to allow inserting dynamic chat history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ---------------------------------------------
# 1. CREATE THE CHAT TEMPLATE
# ---------------------------------------------

# ChatPromptTemplate lets you define a structured prompt containing:
# - system message
# - dynamic chat history
# - human message
chat_template = ChatPromptTemplate([
    
    # A system message to set the behavior of the assistant
    ('system', 'You are a helpful customer support agent'),
    
    # Placeholder to insert the full chat history dynamically
    MessagesPlaceholder(variable_name='chat_history'),
    
    # The final human message in this turn
    ('human', '{query}')
])


# ---------------------------------------------
# 2. LOAD CHAT HISTORY FROM FILE
# ---------------------------------------------

# chat_history will store the conversation so far
chat_history = []

# Open the file containing previous messages (one message per line)
with open('chat_hist.txt') as f:
    
    # readlines() loads all lines from the file into a list
    # extend() adds them to chat_history list
    chat_history.extend(f.readlines())

# Show the loaded chat history
print(chat_history)


# ---------------------------------------------
# 3. FILL THE TEMPLATE WITH REAL VALUES
# ---------------------------------------------

# invoke() fills placeholders inside the ChatPromptTemplate:
# - chat_history → replaces MessagesPlaceholder
# - query → fills the human input {query}
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund'
})

# print the generated prompt (now fully expanded)
print(prompt)
