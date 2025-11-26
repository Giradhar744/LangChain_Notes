from langchain_text_splitters import CharacterTextSplitter

text = '''
Life often moves at a pace faster than we expect, pulling us into routines filled with deadlines, responsibilities, and endless screens. Yet, beyond this constant motion lies a world that invites us to pause and reset. Imagine standing on an open hillside at sunset—the sky slowly shifting from gold to deep orange, birds returning to their nests, and a cool breeze brushing against your skin. In that moment, time feels softer, almost suspended, and the noise of the world seems far away. Such simple experiences remind us that peace doesn’t always come from achieving more, but from allowing ourselves the space to breathe, observe, and reconnect with the world around us. These quiet moments, though often overlooked, have a powerful way of grounding us, helping us rediscover balance and clarity in a life that rarely stops moving.
'''
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap= 0,
    separator='' # '' means Nothing
)

result = splitter.split_text(text)

# for splitting text from documnets which we will get from document_loaders, we use the .split_document() 

print(result)