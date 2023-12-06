from langchain.vectorstores import Chroma
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
persist_directory = 'docs/chroma/'

##Similarity Search

#embedding = OpenAIEmbeddings()
embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                                      model_kwargs={"device": "cpu"})
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Tell me about all-white mushrooms with large fruiting bodies"

smalldb.similarity_search(question, k=2)

smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)


#Maximum marginal relevance
#NOT choose similar responses (most diverse)

question = "what did they say about NIST?"
docs_ss = vectordb.similarity_search(question,k=3)

docs_ss[0].page_content[:100]
docs_ss[1].page_content[:100]

#Adding MMR
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
docs_mmr[0].page_content[:100]
docs_mmr[1].page_content[:100]

### Addressing Specificity: working with metadata
#To address this, many vectorstores support operations on `metadata`.
#`metadata` provides context for each embedded chunk.

question = "what about NIST"

docs = vectordb.similarity_search(
    question,
    k=3,
    filter={'page': 21}
)

for d in docs:
    print(d.metadata)

##Addressing Specificity: working with metadata using self-query retriever
#pip install lark
#from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be on '/home/cmarquez/PycharmProjects/RAQ_langchain/docs/OWASP Application Security Verification Standard 4.0.3-en.pdf'",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the document",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
#llm = OpenAI(temperature=0)

llm = Ollama(model="mistral",
             #model="llama2",
             temperature=0.0,
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "what did they say about XXE?"
docs = retriever.get_relevant_documents(question)

for d in docs:
    print(d.metadata)


##compression
#Information most relevant to a query may be buried in a document with a lot of irrelevant text.
#Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
#Contextual compression is meant to fix this.

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Wrap our vectorstore
#llm = OpenAI(temperature=0)
llm = Ollama(model="mistral",
             #model="llama2",
             temperature=0.0,
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

question = "what did they say about cross site scripting?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)