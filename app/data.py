from dotenv import load_dotenv
import os 
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader


def get_document_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages: 
        text = page.extract_text()
        full_text += text 
    #Now wrap the text into a document to have page_content
    return [Document(page_content=full_text)]

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

