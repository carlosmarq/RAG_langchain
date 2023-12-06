from langchain.vectorstores import Chroma
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

persist_directory = 'docs/chroma/'

#embedding = OpenAIEmbeddings()
embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                                      model_kwargs={"device": "cpu"})

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print(vectordb._collection.count())

#question = "What are major topics for this standard?"
#docs = vectordb.similarity_search(question,k=3)
#len(docs)

#from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(model_name=llm_name, temperature=0)

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="mistral",
             #model="llama2",
             temperature=0.0,
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

#result = qa_chain({"query": question})
#result["result"]


#### prompt

from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Which are the goals of ASVS?"
result = qa_chain({"query": question})
result["result"]
result["source_documents"][0]


import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY" # replace dots with your api key

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]


#Refine:
#Ask the llm for each document, and then sumarise all responses

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
result["result"]