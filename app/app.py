from dotenv import load_dotenv
import os 
load_dotenv()

# Set USER_AGENT for web requests
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs )
    return splitDocs

def create_db(docs):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    vectorStore = FAISS.from_documents(docs, embeddings)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        
        )
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}                                        
    """)

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,

    )

    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain
    

docs = get_document_from_web('https://www.youtube.com/watch?v=-Ueh5XBpcoY&t=682s')
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


response = chain.invoke({
    'input': "What is that?"
})
print(response['answer'])

