import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA  # For question answering
from commons import init_model, init_embedding
from langchain_text_splitters import TokenTextSplitter
from utils import get_pdf_text, get_vectorstore, get_summary, generate_questions

model = init_model()  # Initialize the model from commons.py
embedding = init_embedding()  # Initialize the embedding from commons.py
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini")

def main():
    
    st.set_page_config(page_title="Chatbot", page_icon=":books:")
    st.header("AI Study Buddy")

    with st.sidebar:
        st.subheader("Your documents")
        st.write("Upload your PDFs here")
        pdf_docs = st.file_uploader(" ", accept_multiple_files=True)
        
        
        process_button_disabled = not pdf_docs  # Disable if no PDFs uploaded
        if st.button("Process", disabled=process_button_disabled):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = text_splitter.split_text(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.success("Documents processed!")
                st.session_state.text_chunks = text_chunks # Store text chunks
                
         # Button container for Generate buttons
        st.container()
        col1, col2 = st.columns(2)  # 2 columns for Generate buttons

        get_summary_button_disabled = "text_chunks" not in st.session_state # Disable if no processing
        if col1.button("Generate Summary", key="get_summary_button", disabled=get_summary_button_disabled) and "text_chunks" in st.session_state: # New button and check
            with st.spinner("Generating Summary..."):
                try:
                    summary = get_summary(st.session_state.text_chunks, model) #Use stored text_chunks
                    st.session_state.summary = summary
                    st.success("Summary Generated!")
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")
        
        get_questions_button_disabled  = "text_chunks" not in st.session_state
        if col2.button("Generate Questions", key="get_questions_button", disabled=get_questions_button_disabled) and "text_chunks" in st.session_state:
            with st.spinner("Generating Questions..."):
                try:
                    questions = generate_questions(st.session_state.text_chunks, model)
                    st.session_state.questions = questions  # Store in session state
                    st.success("Questions Generated!")
                except Exception as e:
                    st.error(f"An error occurred during question generation: {e}")

        show_summary_button_disabled = "summary" not in st.session_state
        if col1.button("Show Summary", key="show_summary_button", disabled=show_summary_button_disabled) and "summary" in st.session_state:
            st.session_state.show_summary_popup = True
            
        show_questions_button_disabled = "questions" not in st.session_state
        if col2.button("Show Questions", key="show_questions_button", disabled=show_questions_button_disabled) and "questions" in st.session_state:
            st.session_state.show_questions = True
    
    if "show_summary_popup" in st.session_state and st.session_state.show_summary_popup:  # Check the flag
            st.write(st.session_state.summary)
            
            if st.button("Close Summary", key="close_summary_button"):  # Add a close button
                st.session_state.show_summary_popup = False  # Reset the flag


    if "show_questions" in st.session_state and st.session_state.show_questions:
        st.subheader("Generated Questions")
        for question in st.session_state.questions:
            st.write(f"- {question}")
            
        if st.button("Close Questions", key="close_questions_button"):  # Add a close button for questions
            st.session_state.show_questions = False
    
    
   # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for the user
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process user input and generate assistant's response
        if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
            vectorstore = st.session_state.vectorstore
            qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=vectorstore.as_retriever())
            with st.spinner("Generating Answer"):
                answer = qa_chain.run(user_input)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
        else:
            st.warning("Please process the documents first.")



if __name__ == "__main__":
    main()
   
        
