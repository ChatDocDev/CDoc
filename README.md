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

* Python >=3.9 (https://www.python.org/downloads/)
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

4. Install dependencies from requirements.txt
   ```
   pip install -r requirements.txt
   ```

5. Pull the required models from Ollama
   
   - Download & install [Ollama](https://ollama.com/) if not installed
   - Open terminal & run these command to pull the required models into local machine
     
     For `llama3`
     ```
     ollama pull llama3:latest
     ```    
     For `nomic-embed-text`
     ```
     ollama pull nomic-embed-text:latest
     ``` 
   - Insure both models are downloaded
     ```
     ollama ls
     ```
      <p align=center>
      <img width="70%" alt="Screenshot 2024-08-26 at 12 36 17 PM" src="https://github.com/user-attachments/assets/d88e532f-f679-471d-a876-6fc9b0d93ab2">
      </p>
   - Serve Ollama
     ```
     ollama serve
     ```
     goto `localhost:11434` & you should get `Ollama is running`
     <p align=center>
     <img width="612" alt="Screenshot 2024-08-26 at 12 59 48 PM" src="https://github.com/user-attachments/assets/38f139fe-753a-40cc-81c6-a6d178d8137f">
     </p>

7. BACKEND

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
   <p align="center">
   <img width="90%" alt="Screenshot 2024-08-26 at 1 12 53 PM" src="https://github.com/user-attachments/assets/60265a02-5006-4788-9ddb-8afb2e82371e">
   </p>

   Open new terminal and go into backend folder(hint: `cd backend`) & Run backend server:
   ```
   python backend.py
   ```
   <p align="center">
   <img width="90%" alt="Screenshot 2024-08-26 at 1 23 53 PM" src="https://github.com/user-attachments/assets/5cd1d4ac-c1a3-48f3-abca-834934950226">
   </p>
   
8. FRONTEND

   Open new terminal and go to frontend folder
   ```
   cd frontend
   ```

   Run frontend.py
   ```
   streamlit run frontend.py
   ```
   <p align="center">
   <img width="90%" alt="Screenshot 2024-08-26 at 1 26 54 PM" src="https://github.com/user-attachments/assets/d6d7799c-47f7-455f-a7e0-9d561c9db96f">
   </p>


<p align ="center">
   <img width="100%" alt="Screenshot 2024-08-26 at 1 30 19 PM" src="https://github.com/user-attachments/assets/8752ed5d-44a8-498d-a857-aec9c2e76258">

</p>
