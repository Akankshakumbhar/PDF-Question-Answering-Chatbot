
 📚 PDF Q&A Chatbot using LangChain + Streamlit

This is a PDF-based Question-Answering Chatbot that allows users to query multiple predefined PDF documents. It uses **LangChain**, **FAISS**, **Ollama embeddings**, and a local LLM (**Gemma:7B**) to extract answers based on the document content. The frontend is built using **Streamlit**.
 🔍 Features

- ✅ Ask questions directly from the content of predefined PDF files  
- ✅ Retrieval-Augmented Generation (RAG) pipeline  
- ✅ PDF text extraction and intelligent chunking  
- ✅ Semantic similarity-based retrieval using FAISS  
- ✅ Local LLM (`gemma:7b`) used for response generation  
- ✅ Responses include page numbers and document names (if possible)  
- ✅ Session-based chat history using Streamlit state

 🧰 Tech Stack

| Component         | Technology Used                            |
|------------------|---------------------------------------------|
| Frontend         | Streamlit                                   |
| LLM              | Ollama with `gemma:7b`                      |
| Embeddings       | `nomic-embed-text` via Ollama               |
| PDF Loader       | PyPDFLoader (LangChain)                     |
| Vector Store     | FAISS                                       |
| Chunking         | RecursiveCharacterTextSplitter              |
| Prompting        | LangChain LLMChain with custom prompt       |

🚀 How It Works

1. **PDFs are loaded**: The app loads predefined PDF files.
2. **Text Splitting**: Documents are split into manageable chunks.
3. **Embedding & Indexing**: Chunks are converted into vectors and stored in a FAISS index.
4. **Question Input**: User submits a question via the interface.
5. **Semantic Search**: Retrieves the most relevant chunks from the vector database.
6. **LLM Generation**: A local LLM (Gemma 7B) generates an answer based on the context.
7. **Chat History**: Maintains the full session-based conversation.

 📂 Project Structure

├── streamlitlatest.py       # Main Streamlit App
├── qpm help.pdf             # Sample PDFs
├── microservice arcticture.pdf
├── orcerstractionframework.pdf
├── startupai-financial-report-v2.pdf
└── faiss\_index/             # Saved FAISS vector index

 ⚙️ Setup Instructions
. Create a virtual environment & install dependencies

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
streamlit
langchain
langchain-community
faiss-cpu
pypdf2
PyMuPDF
ollama
```

### 3. Make sure you have Ollama installed and models pulled

Install Ollama: [https://ollama.com/download](https://ollama.com/download)

Then run:

```bash
ollama pull gemma:7b
ollama pull nomic-embed-text
```

### 4. Run the Streamlit app

```bash
streamlit run streamlitlatest.py

## 🖼️ Example UI Flow

* Title: 📚 PDF Q\&A Chatbot
* Input: Ask a question (e.g., "What is microservice architecture?")
* Output: Bot provides context-based answer with page number and document name
* Chat history is visible and scrollable

📌 Notes

* Only the predefined PDF files are used (edit the file list in the script to change this).
* You can extend it to support **custom file uploads**.
* Local model ensures privacy and control over responses.

## 💡 Future Enhancements

* Enable uploading PDFs via UI
* Multi-language support
* Streamlit Cloud or Hugging Face deployment
* Database-backed chat memory
* PDF content summarization feature

 👤 Author

**Akanksha Kumbhar**
MSc Computer Science | AI & ML Enthusiast | Python Developer


