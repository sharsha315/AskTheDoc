from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain import hub

def load_and_split_pdf(file_path):
    """Loads and splits the PDF document into smaller chunks."""
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    return docs

def process_docs(docs):
    """Splits, embeds, creates a vector store, and sets up retriever."""
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    
    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/text-embedding-004")
    
    # Create vector store
    vector_stores = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    # Return retriever from vector store
    retriever = vector_stores.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    return retriever

def get_llm():
    """Returns the Google Generative AI language model."""
    llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-pro", temperature=0.3)
    return llm

def get_rag_chain(llm, retriever):
    """Sets up the Retrieval-Augmented Generation (RAG) chain."""
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return rag_chain

def get_answer(rag_chain, query):
    """Retrieves the answer for the user's query using the RAG chain."""
    answer = rag_chain.invoke({"query": query})
    return answer['result']

def main():
    """Main function that handles the flow of the RAG search."""
    file_path = "/workspace/AskTheDoc/AZ-900 Notes.pdf"  # Example file path
    query = "What are the types of service available on Microsoft Azure?"

    # Load the PDF
    docs = load_and_split_pdf(file_path)

    # Process the PDF, get retriever, and set up the LLM
    retriever = process_docs(docs)
    llm = get_llm()

    # Create RAG chain and retrieve the answer
    rag_chain = get_rag_chain(llm, retriever)
    answer = get_answer(rag_chain, query)

    # Print the result
    print(answer)

if __name__ == "__main__":
    main()