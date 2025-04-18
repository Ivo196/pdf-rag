from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector_db import vectorStore



def create_chain():
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

    retriever = vectorStore().as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

def chat_ai(message):
    
    chain = create_chain()
    response = chain.invoke({
        'input': message
    })
    return(response['answer'])
