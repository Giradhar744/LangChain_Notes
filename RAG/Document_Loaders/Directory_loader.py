from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path= 'C:/Users/girdh/Desktop/Langchain/RAG/Document_Loaders/Software_Eng',
    glob= '*.pdf',
    loader_cls= PyPDFLoader # type: ignore
)

# for small files and data
# docs = loader.load()
# print(docs[50].page_content)
# print(docs[50].metadata)


# For big files and data
docs = loader.lazy_load()

for documents in docs:
    print(documents.metadata)