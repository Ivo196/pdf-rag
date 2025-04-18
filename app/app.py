from dotenv import load_dotenv
import os 
load_dotenv()

from chain import create_chain   

# Set USER_AGENT for web requests
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'


def chat_ai(message):
    
    chain = create_chain()
    response = chain.invoke({
        'input': message
    })
    return(response['answer'])

