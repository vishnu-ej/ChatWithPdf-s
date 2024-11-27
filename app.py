import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name="Llama3-8b-8192"
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        input_key='question',
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

def format_message_content(message):
    if isinstance(message, HumanMessage):
        return f"**User:** {message.content}"
    elif isinstance(message, AIMessage):
        return f"**Assistant:** {message.content}"
    else:
        return f"**System:** {message.content}"

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process documents first!")
        return

    try:
        # Get the response
        response = st.session_state.conversation({
            "question": user_question
        })
        
        # Extract answer and source documents
        answer = response.get('answer', "No answer generated")
        source_docs = response.get('source_documents', [])

        # Display the current question and answer
        st.markdown(f"**User:** {user_question}")
        st.markdown(f"**Assistant:** {answer}")
        st.markdown("---")

        # Display chat history if it exists
        if 'chat_history' in response:
            st.subheader("Previous Conversation")
            for message in response['chat_history']:
                st.markdown(format_message_content(message))
                st.markdown("---")

        # Display source documents in an expander
        with st.expander("View Source Documents", expanded=False):
            st.markdown("### Retrieved Context")
            if source_docs:
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Document {i+1}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
            else:
                st.warning("No source documents were retrieved.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try rephrasing your question or uploading the documents again.")

def main():
    st.set_page_config(
        page_title="RAG Chat System",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Document Chat Assistant")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document.")
                return
                
            with st.spinner("Processing documents..."):
                try:
                    # Get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Created {len(text_chunks)} text chunks")
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Created vector store")
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

    # Main chat interface
    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.info("Please upload and process documents to start chatting.")

if __name__ == "__main__":
    main()