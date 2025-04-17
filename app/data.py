from dotenv import load_dotenv
import os 
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



def get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


def splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs )
    return splitDocs