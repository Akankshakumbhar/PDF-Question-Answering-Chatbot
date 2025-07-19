'''from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_pdf.pdf import PyPDFLoader


# from langchain.chains import RetrievalQA  # We'll handle the LLM part separately for multilingual output
from langchain_ollama import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Update this import
from langchain_ollama import OllamaLLM

# Update the code to use OllamaLLM instead of Ollama
#llm = OllamaLLM(model="gemma:7b")  # Choose a suitable LLM


# List of PDF files to load
pdf_files = ["qpm help.pdf", "microservice arcticture.pdf", "orcerstractionframework.pdf"]  # Add more PDFs as needed

# Initialize an empty list to store all documents
all_docs = []

# Loop through each PDF file and load its content
for pdf in pdf_files:
    try:
        loader = PyPDFLoader(pdf)
        docs = loader.load()  # Load pages from the PDF
        all_docs.extend(docs)  # Append them to the main list
    except Exception as e:
        print(f"Error loading PDF {pdf}: {e}")

# Check loaded documents
print(f"Total pages loaded: {len(all_docs)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
splited = text_splitter.split_documents(all_docs)
# print(splited)

from langchain_text_splitters import CharacterTextSplitter

text_split = CharacterTextSplitter(separator='\n \n', chunk_size=5000, chunk_overlap=500)
split1 = text_split.split_documents(all_docs)
# print(split1)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(documents=splited, embedding=embeddings)

db
print(" FAISS vector store created successfully")

# print(db)

db.save_local("faiss_index")
query = "Getting started with QPM application"
ans = db.similarity_search(query)
# print(ans[0].page_content)

retriever = db.as_retriever()
# res=reteriver.invoke(query3)
# print(res[0].page_content)

vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize the LLM
llm = Ollama(model="gemma:7b")  # Choose a suitable LLM

# Define the prompt for multilingual output
prompt_template = """You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer in the same language as the question:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

while True:
    query = input("Enter your query (or type 'exit' to stop): ")

    if query.lower() == "exit":
        print("Exiting the search.")
        break  # Stop the loop

    results = db.similarity_search(query)

    if results:
        context = "\n".join([doc.page_content for doc in results])
        response = llm_chain.run(context=context, question=query)
        print("\n Answer:", response, "\n")
    else:
        print("\n No relevant content found.\n")


while True:
    query = input("Enter your query (or type 'exit' to stop): ")
    
    if query.lower() == "exit":
        print("Exiting the search.")
        break  # Stop the loop

    results = db.similarity_search(query)
    
    if results:
        print("\n Answer:", results[0].page_content, "\n")
    else:
        print("\n No relevant content found.\n")'''



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# List of PDF files to load
pdf_files = ["qpm help.pdf", "microservice arcticture.pdf", "orcerstractionframework.pdf"]

# Initialize an empty list to store all documents
all_docs = []

# Loop through each PDF file and load its content
for pdf in pdf_files:
    try:
        loader = PyPDFLoader(pdf)
        docs = loader.load()  # Load pages from the PDF
        all_docs.extend(docs)  # Append them to the main list
    except Exception as e:
        print(f"Error loading PDF {pdf}: {e}")

# Check loaded documents
print(f"Total pages loaded: {len(all_docs)}")

# Split the documents into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
splited = text_splitter.split_documents(all_docs)

# Create embeddings using OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create the FAISS vector store from the documents
db = FAISS.from_documents(documents=splited, embedding=embeddings)

print("FAISS vector store created successfully")

# Save the FAISS index for future use
db.save_local("faiss_index")

# Load the FAISS vector store (just to be sure)
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Query loop
while True:
    query = input("Enter your query (or type 'exit' to stop): ")
    
    if query.lower() == "exit":
        print("Exiting the search.")
        break  # Stop the loop

    # Retrieve relevant answer from the FAISS database
    results = db.similarity_search(query)
    
    if results:
        print("\n Answer:", results[0].page_content, "\n")
    else:
        print("\n No relevant content found.\n")
