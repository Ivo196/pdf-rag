from dotenv import load_dotenv
import os 
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

def get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

docs = get_document_from_web('https://python.langchain.com/docs/how_to/document_loader_web/')

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    
    )
prompt = ChatPromptTemplate.from_template("""
Answer the user's question:
Context: {context}
Question: {input}                                        
""")
chain = prompt | model

response = chain.invoke({
    'input': "What is that?",
    'context': docs
})
print(response)

