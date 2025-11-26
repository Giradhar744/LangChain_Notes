from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('C:/Users/girdh/Desktop/Langchain/RAG/Document_Loaders/Giradhar_Gopal_Resume (2).pdf')
docs = loader.load()

print(docs[0])