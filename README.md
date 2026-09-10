# Enterprise RAG Assistant

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions using natural language. The system retrieves relevant information from the uploaded documents and generates contextual responses with citations.

## Features

- PDF document upload and processing
- Semantic search for relevant document content
- AI-powered question answering using RAG
- Citation-based responses
- User authentication
- Chat history
- Dockerized application

## How It Works

```text
User
  ↓
Upload PDF
  ↓
Document Processing
  ↓
Text Extraction & Chunking
  ↓
Embeddings
  ↓
Vector Store
  ↓
Semantic Retrieval
  ↓
Relevant Context
  ↓
LLM
  ↓
Answer + Citations
```

## Technology Stack

### Backend
- Python
- FastAPI

### AI / LLM
- LangChain
- RAG
- Embeddings
- Large Language Model

### Database / Storage
- PostgreSQL
- Vector Database

### Development Tools
- Git
- GitHub
- Postman

## Project Structure

```text
Enterprise-RAG-Assistant/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Enterprise-RAG-Assistant.git
cd Enterprise-RAG-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

```env
LLM_API_KEY=your_api_key
DATABASE_URL=your_database_url
```


### 5. Run the application

```bash
uvicorn app.main:app --reload
```

Open the application in your browser at:

```text
http://127.0.0.1:8000
```

## Running with Docker

Build the Docker image:

```bash
docker build -t enterprise-rag-assistant .
```

Run the container:

```bash
docker run -p 8000:8000 enterprise-rag-assistant
```

Or use Docker Compose:

```bash
docker compose up --build
```

## Application Workflow

1. User authenticates with the application.
2. User uploads a PDF document.
3. The application extracts and processes the document content.
4. Document content is divided into smaller chunks.
5. Embeddings are generated for the processed content.
6. Relevant chunks are retrieved based on the user's question.
7. Retrieved context is provided to the LLM.
8. The LLM generates an answer based on the retrieved information.
9. Relevant citations are displayed with the response.
10. Conversation history is maintained for the user.

## Use Cases

The application can be adapted for:

- Enterprise knowledge management
- HR policy assistants
- Legal document analysis
- Research document search
- Technical documentation assistants
- Internal company knowledge bases

## Security

- Environment variables are used for sensitive configuration.
- API keys are excluded from version control.
- User authentication protects application access.
- `.gitignore` prevents sensitive and unnecessary files from being committed.

## Future Enhancements

- Role-based access control
- Support for multiple document formats
- Advanced RAG evaluation
- Conversation-based document retrieval
- Document management dashboard
- CI/CD pipeline
- Production monitoring
- Improved document-level access control

## Author

**ELE Ganesh**

GitHub: https://github.com/Ganesh0Ele