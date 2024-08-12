# CDoc: Chat with Your Document

CDoc empowers you to have a conversation with your documents using local large language models (LLMs) and the power of Ollama, ChromaDB, and LangChain.

**Key Features:**

* **Chat with Documents:** Ask questions and get answers directly from your documents.
* **Local LLM Support:** Leverage the capabilities of local LLMs for offline document interaction.
* **ChromaDB Support:** Store and manage document metadata efficiently with ChromaDB.
* **LangChain Integration:** Streamline information extraction from documents through LangChain.

**Target Users:**

* Researchers and students seeking an efficient way to interact with research papers.
* Developers and programmers looking to analyze code documentation.
* Professionals wanting to extract key information from contracts and legal documents (Optional with OpenAI API).

## Installation

**Prerequisites:**

* Python 3.x (https://www.python.org/downloads/)
* pip (usually comes pre-installed with Python)
* Ollama (https://ollama.com/)

**Installation Steps:**

1. **Clone the repository:**

   ```
   git clone https://github.com/ChatDocDev/CDoc
   ```
2. **Navigate to the project directory:**

```
cd CDoc
```

3. Open project directory in VSCode

   ```
   code .
   ```

   or any other code editor
4. BACKEND

   go to `backend` directory

   ```
   cd backend
   ```

   create `db` folder for storing Chromadb files

   ```
   mkdir db
   ```

   Start Chromadb server:

   ```
   chroma run --path db --port 8001
   ```

   Open new terminal and go into backend folder(hint: `cd backend`) & Run backend server:

   ```
   python backend.py
   ```
5. FRONTEND

   Open new terminal and go to frontend folder

   ```
   cd frontend
   ```

    Run frontend.py

```
    streamlit run frontend.py
```
