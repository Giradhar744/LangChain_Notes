from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity  # the input for the cosine similarity  to be a 2D list.
import numpy as np


Embedding_model = HuggingFaceEmbeddings(model = 'sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "Virat Kohli Modern batting great known for aggressive play and world-class chasing.",
    "Rohit Sharma Elegant opener and only player with three ODI double centuries.",
    "MS Dhoni Legendary captain who won all ICC trophies; known for calm finishing.",
    "Sachin Tendulkar ‘God of Cricket’ with 100 international centuries.",
    "Jasprit Bumrah Premier fast bowler famous for deadly yorkers and unique action.",
    "Hardik Pandya Powerful all-rounder with explosive batting and medium-fast bowling.",
    "Yuvraj Singh 2011 WC hero; famous for hitting 6 sixes in an over.",
    "Shubman Gill Young batting star known for timing and consistency.",
    "Rishabh Pant Fearless wicketkeeper-batsman with match-winning Test knocks.",
    "AB de Villiers ‘Mr. 360°’; innovative and destructive batsman.",
    "Kane Williamson Technically gifted NZ batsman and calm leader.",
    "Ben Stokes Impact all-rounder known for heroic match-saving performances.",
    "Joe Root Elegant Test batsman; part of modern cricket’s Fab Four.",
    "Steve Smith Unorthodox but highly effective Test batsman.",
    "Pat Cummins World-class fast bowler and captain; led Australia to major titles."
]

docs_embedding =  Embedding_model.embed_documents(docs)

query = 'Tell me about rohit sharma'

query_embedding = Embedding_model.embed_query(query)

scores = cosine_similarity([query_embedding], docs_embedding)[0]  #type:ignore

index , score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]  # enmureate is used here to attach the index or a unique number to each score which is index in this case.


print(query)
print(docs[index])
print("Similarity Score is:",score)

