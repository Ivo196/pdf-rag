from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings


pc = Pinecone()
index = pc.Index('langchain')

def vectorStore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    vectorStore = PineconeVectorStore(
        embedding=embeddings,
        index=index
    )
    #vectorStore.add_documents(docs)
    return vectorStore

