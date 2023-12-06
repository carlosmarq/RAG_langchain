from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

persist_directory = 'docs/chroma/'
embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                          model_kwargs={"device": "cpu"})
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

question = "What is the topic of the document?"
#docs = vectordb.similarity_search(question,k=3)
#len(docs)

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="mistral",
             #model="llama2",
             temperature=0.0,
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
#llm.predict("Hello world!")

# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
question = "What is XSS"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


#result = qa_chain({"query": question})
#result["result"]

#Memory

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "What is an HSM?"
print(question)
result = qa({"question": question})
result['answer']

question = "when can it be used?"
print(question)
result = qa({"question": question})
result['answer']