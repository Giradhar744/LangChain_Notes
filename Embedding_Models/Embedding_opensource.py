from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')
# It returns 384 dimension embedding vector
# text = 'Delhi is the capital of India'

# vector = embedding.embed_query(text)
# print(vector)

documents = [
    "Delhi is capital of india",
    "Kolkata is the capital of bengal",
    "paris is the capital of france"
]
result = embedding.embed_documents(documents)
print(result)
