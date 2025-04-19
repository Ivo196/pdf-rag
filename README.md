# PDF RAG Assistant

A Streamlit application that allows users to interact with PDF documents and web content using Retrieval Augmented Generation (RAG) technology. The application enables users to upload PDF files or provide URLs, and then ask questions about the content.

## Features

- Upload and process PDF documents
- Load and process content from web URLs
- Interactive chat interface for asking questions
- Retrieval Augmented Generation (RAG) for accurate answers
- Document chunking and vector storage
- Clear chat history functionality

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd pdf-rag
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run app/app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the application:
   - Upload a PDF file using the sidebar
   - Or enter a URL to process web content
   - Once the content is processed, you can start asking questions
   - Use the chat interface to interact with the content

## Project Structure

```
pdf-rag/
├── app/
│   ├── app.py           # Main Streamlit application
│   ├── chain.py         # RAG chain implementation
│   ├── data.py          # Document processing functions
│   └── vector_db.py     # Vector database functionality
├── .env                 # Environment variables
└── README.md           # Project documentation
```

## Dependencies

- streamlit
- langchain
- langchain-openai
- langchain-pinecone
- pinecone-client
- PyPDF2
- python-dotenv

## License

[Your chosen license]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
