import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("📚 PDF Q&A Chatbot")
st.markdown("Ask questions about the predefined documents.")

# Load vector DB (only once)
@st.cache_resource
def load_vectorstore():
    pdf_files = ["qpm help.pdf", "microservice arcticture.pdf", "orcerstractionframework.pdf","startupai-financial-report-v2.pdf"]
    all_docs = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error loading PDF {pdf}: {e}")
    #st.write(f"✅ Loaded {len(all_docs)} pages from PDFs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    splited_docs = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(documents=splited_docs, embedding=embeddings)
    db.save_local("faiss_index")

    return db

# Load the vector DB and initialize LLM + chain
vector_db = load_vectorstore()
retriever = vector_db.as_retriever()

llm = Ollama(model="gemma:7b")
prompt_template = """You are a helpful assistant. Answer the question based on the provided context. 
Also, please include the **page number(s)** and the **document name** if possible.
 If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer in the same language as the question:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Chat input
query = st.text_input("Enter your question:", placeholder="Ask about QPM, microservices, orchestration...")

if query:
    # Run similarity search
    results = retriever.get_relevant_documents(query)
    if results:
        context = "\n".join([doc.page_content for doc in results])
        answer = llm_chain.run(context=context, question=query)
    else:
        answer = "No relevant content found."

    # Update chat history
    st.session_state.chat_history.append({"question": query, "answer": answer})

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")
