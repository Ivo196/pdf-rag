from dotenv import load_dotenv
import os 
load_dotenv()

# Set USER_AGENT for web requests
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'


from data import get_document_from_web, splitter
from vector_db import create_db
from chain import create_chain   

docs = get_document_from_web('https://www.youtube.com/watch?v=-Ueh5XBpcoY&t=682s')
split_docs = splitter(docs)
vectorStore = create_db(split_docs)
chain = create_chain(vectorStore)

def chat_ai(message):
    response = chain.invoke({
        'input': message
    })
    return(response['answer'])

