from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')
text = '''
Environmental conservation is gaining importance as climate change continues to affect weather patterns and global temperatures. Renewable energy sources like solar and wind power are helping reduce our dependence on fossil fuels. Additionally, wildlife protection efforts aim to preserve endangered species and maintain ecological balance. Waste management practices, such as recycling and composting, also play a crucial role in keeping the environment clean and sustainable.

Modern technology has transformed the way we communicate, especially through smartphones that allow instant messaging and video calls across the world. At the same time, artificial intelligence is revolutionizing industries by automating tasks, improving predictions, and enhancing decision-making. Meanwhile, cybersecurity has become a major concern, as increased online activity requires stronger protection against hackers and data leaks.
'''
splitter = SemanticChunker(
    embedding, breakpoint_threshold_type = 'standard_deviation',
    breakpoint_threshold_amount =1
)
   

result = splitter.create_documents([text])

print(len(result))
print(result)