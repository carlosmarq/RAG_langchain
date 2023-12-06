#!pip install sentence_transformers InstructorEmbedding
#https://huggingface.co/hkunlp/instructor-xl
#https://huggingface.co/hkunlp/instructor-base
#https://huggingface.co/hkunlp

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

#from langchain.embeddings.openai import OpenAIEmbeddings
#embedding = OpenAIEmbeddings()


embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                                      model_kwargs={"device": "cpu"})

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)


import numpy as np

np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
np.dot(embedding2, embedding3)



from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("/home/cmarquez/PycharmProjects/RAQ_langchain/docs/OWASP Application Security Verification Standard 4.0.3-en.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

    # Split
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

splits = text_splitter.split_documents(docs)
len(splits)

from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
#!rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

#semantic similarity

question = "who are the authors of the document"
docs = vectordb.similarity_search(question,k=3)
len(docs)
docs[0].page_content


vectordb.persist()