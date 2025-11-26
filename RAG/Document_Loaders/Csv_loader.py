from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path= 'C:/Users/girdh/Desktop/Langchain/RAG/Document_Loaders/my_csv_file.csv'
)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)