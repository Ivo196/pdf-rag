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
    You are a helpful AI assistant that answers questions based on the provided context.
    
    Context: {context}
    
    Instructions for using the context:
    1. Carefully analyze the provided context to find relevant information
    2. If the context contains the answer, use it directly
    3. If the context doesn't contain enough information, say so clearly
    4. If the context contains multiple relevant pieces, combine them coherently
    5. Always cite specific parts of the context when possible
    6. If the question is unclear or ambiguous, ask for clarification
    
    Question: {input}
    
    Please provide a clear, concise, and accurate answer based on the context above.
    If you're unsure about any part of the answer, acknowledge the uncertainty.
    """)

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,

    )

    retriever = vectorStore().as_retriever(search_kwargs={'k': 3})
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
