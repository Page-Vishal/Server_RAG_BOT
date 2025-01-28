
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

api_key = "Your_API_KEY" 

# Load and split PDF documents
def initialize_vector_store():
    loader = PyPDFLoader("NN.pdf")  # Ensure 'Data' directory exists with PDFs
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

vector_store = initialize_vector_store()

'''
# Basic Example (no streaming)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token= os.environ["HUGGINGFACEHUB_API_TOKEN"]
)
'''
#print(llm.invoke("What is Deep Learning?"))

GROQ_LLM = ChatGroq(
    api_key = api_key,
    model = 'gemma2-9b-it'
)

# Load LLM and process the query
def answer(query):
    try:
        
        chain = RetrievalQA.from_chain_type(
            llm=GROQ_LLM,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 1})
        )
        
        result = chain.invoke(query)  # ({"query": query}, return_only_outputs=True)
        return result
    except Exception as e:
        return {"error": str(e)}
