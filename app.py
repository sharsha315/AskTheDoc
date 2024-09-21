import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

def load_pdf_doc(pdf_file):
    reader = PdfReader(pdf_file)
    documents = ""
    for page in reader.pages:
        #page = reader.pages[page_num]
        documents += page.extract_text()
    
    return documents


def load_url_doc(url):
    return [{"page_content": f"Text extracted from URL: {url}"}]  # Placeholder for URL text

def split_documents(documents, chunk_size=10000, chunk_overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.create_documents(documents)
    return texts

# Function to embed and store documents in a vector store
def embed_and_store(docs, embeddings):
    texts = [doc.page_content for doc in docs]
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    return vector_store

# Initialize GoogleGenerativeAI LLM and embeddings
def initialize_llm(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model = "models/embedding-001")
    llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro", temperature=0.3)
    return llm, embeddings

# Process PDF/URL input and create vector store
def process_rag_source(source_type, pdf_file=None, url=None, embeddings=None):
    documents = []

    # Get documents based on the source type
    if source_type == "PDF" and pdf_file:
        documents = load_pdf_doc(pdf_file)
    elif source_type == "URL" and url:
        documents = load_url_doc(url)
    else:
        st.error("Please upload a PDF or provide a valid URL.")
        return None

    # Split the documents into smaller chunks
    texts = split_documents(documents)

    # Store the chunks in the vector store using embeddings
    vector_store = embed_and_store(texts, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Set up the chatbot with retrieval-based QA
def setup_chatbot(llm, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Streamlit app layout
def main():
    # Sidebar for inputs
    st.sidebar.title("AskTheDoc Options")
    
    # Input Gemini API Key
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

    # Check for API key
    if not api_key:
        st.sidebar.error("Please enter your Gemini API key.")
        return

    genai.configure(api_key=api_key)


    # Source type selection (PDF or URL)
    source_type = st.sidebar.radio("Select the source type:", ["PDF", "URL"])

    # File uploader or URL input based on the source type
    pdf_file, url = None, None
    if source_type == "PDF":
        pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    elif source_type == "URL":
        url = st.sidebar.text_input("Enter URL")

    # Submit button for PDF/URL
    if st.sidebar.button("Submit Options"):
        # Initialize LLM and embeddings
        llm, embeddings = initialize_llm(api_key)
        
        # Process the RAG source
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        if vector_store:
            pass
        else:
            vector_store = process_rag_source(source_type, pdf_file=pdf_file, url=url, embeddings=embeddings)
        
        if vector_store:
            st.sidebar.success(f"{source_type} processed successfully!")

            # Option to ask questions by type or voice
            query_type = st.sidebar.radio("Ask your question by:", ["Type", "Voice"])

            if query_type == "Type":
                question = st.sidebar.text_input("Type your question here")
            elif query_type == "Voice":
                st.sidebar.error("Voice input not implemented yet.")  # Placeholder for voice input handling

            # Submit button for questions
            if st.sidebar.button("Ask Question"):
                if question:
                    # Setup Chatbot
                    
                    qa_chain = setup_chatbot(llm, vector_store)
                    
                    # Ask the question
                    response = qa_chain.invoke(question)
                    st.sidebar.success("Question submitted!")

                    # Main page content: Display question and chatbot response
                    st.title("AskTheDoc - Chatbot")
                    st.write(f"**User Question:** {question}")
                    st.write(f"**Chatbot Response:** {response}")
                else:
                    st.sidebar.error("Please enter a question.")

if __name__ == "__main__":
    main()
