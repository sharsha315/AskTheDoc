import os
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain_community.vectorstores import FAISS
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#import PyPDF2

def extract_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        #page = reader.pages[page_num]
        text += page.extract_text()
    
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    # vector_store.save_local("index")
    db = Chroma.from_documents(text_chunks, embedding=embeddings)

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = get_vector_store(text_chunks)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



def main():
    st.set_page_config("AskTheDoc")
    st.header("AskTheDoc")
    st.subheader("Learn Smartly with Documents using Gemini")
    
    with st.sidebar:
        st.title("AskTheDoc")

        GOOGLE_API_KEY = st.text_input("Enter GOOGLE_API_KEY", type="password")

        source_type = st.radio("Select the source:", ["PDF", "URL"])
        if source_type == "PDF":
            pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        elif source_type == "URL":
            url = st.text_input("Enter URL")

        query_type = st.radio("How do you want to ask your question?", ["Type", "Speak"])
        
        if st.button("Submit"):
            with st.spinner("Processing..."):
                # raw_text = get_pdf_text(pdf_docs)
                # text_chunks = get_text_chunks(raw_text)
                # get_vector_store(text_chunks)
                if source_type == "PDF" and pdf_file:
                    context = extract_from_pdf(pdf_file)
                    text_chunks = get_text_chunks(context)
                    get_vector_store(text_chunks)
               

                
                st.success("Done")

    # query_type = st.radio("How do you want to ask your question?", ["Type", "Speak"])

    if query_type == "Type":
        question = st.text_input("Type your question here")
        if question:
            user_input(question)
    elif query_type == "Speak":
        audio_file = st.file_uploader("Upload your voice query", type=["wav", "mp3"])

    # Process the query
    # if st.button("Submit"):

    #     if source_type == "PDF" and pdf_file:
    #         print()
    #         #context = extract_text_from_pdf(pdf_file)

    #     elif source_type == "URL" and url:
    #         #context = fetch_url_data(url)
    #         print()
        
    #     if query_type == "Type":
    #         # response = query_llm(question, context)
    #         # st.write(response)
    #         # text_to_speech(response)
    #         print()
    #     elif query_type == "Speak" and audio_file:
    #         # question = transcribe_voice(audio_file)
    #         # st.write(f"You said: {question}")
    #         # response = query_llm(question, context)
    #         # st.write(response)
    #         # text_to_speech(response)
    #         print()



if __name__ == "__main__":
    main()