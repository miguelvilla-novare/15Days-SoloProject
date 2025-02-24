import os
import streamlit as st
from display import display_quiz, display_flashcards 
from langchain.chains import RetrievalQA 
from commons import init_model, init_moderation
from langchain_text_splitters import TokenTextSplitter
from actions import (
    generate_and_store_summary, 
    generate_and_store_flashcards, 
    generate_and_store_quiz, 
    is_activity_active, 
    initialize_chat_history, 
    display_chat_history, 
    handle_user_input, 
    process_uploaded_pdfs
)

# Initialize model & text splitter
model = init_model()  
moderation = init_moderation()
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini")

def main():
    st.set_page_config(page_title="Chatbot", page_icon=":books:")
    st.header("AI Study Buddy")

    initialize_chat_history()
    activity_active = is_activity_active()

    # Initialize moderation warning flag
    if "moderation_warning" not in st.session_state:
        st.session_state.moderation_warning = False

    with st.sidebar:
        st.subheader("Your documents")
        st.write("Upload your PDFs here")

        pdf_docs = st.file_uploader(" ", accept_multiple_files=True)
        process_uploaded_pdfs(pdf_docs, text_splitter)

        st.markdown("---")  # Divider for separation
        st.subheader("Actions")
        st.container()
        col1, col2 = st.columns(2)

        # Initialize session state flags
        if "generating_summary" not in st.session_state:
            st.session_state.generating_summary = False
        if "generating_quiz" not in st.session_state:
            st.session_state.generating_quiz = False
        if "generating_flashcards" not in st.session_state:
            st.session_state.generating_flashcards = False

        # Generate Summary Button
        if col1.button("Generate Summary", key="get_summary_button", disabled="text_chunks" not in st.session_state) and "text_chunks" in st.session_state:
            st.session_state.generating_summary = True
            with st.spinner("Generating Summary..."):
                generate_and_store_summary(st.session_state.text_chunks, model)
            st.session_state.generating_summary = False

        # Show Summary Button
        if col2.button("Show Summary", key="show_summary_button", disabled="summary" not in st.session_state) and "summary" in st.session_state:
            st.session_state.show_summary_popup = True

        # Generate Flashcards Button
        if col1.button("Generate Flashcards", key="generate_flashcards_button", disabled="text_chunks" not in st.session_state):
            st.session_state.generating_flashcards = True
            with st.spinner("Generating Flashcards..."):
                generate_and_store_flashcards(st.session_state.text_chunks, model)
            st.session_state.generating_flashcards = False

        # Take Quiz Button
        if col2.button("Take Quiz", key="take_quiz_button", disabled="text_chunks" not in st.session_state):
            st.session_state.generating_quiz = True
            with st.spinner("Generating Quiz..."):
                generate_and_store_quiz(st.session_state.text_chunks, model)
            st.session_state.generating_quiz = False

    # Display quiz if active
    if "quiz_questions" in st.session_state:
        display_quiz()

    # Display flashcards if active
    if "flashcards" in st.session_state:
        display_flashcards()

    # Show summary popup
    if "show_summary_popup" in st.session_state and st.session_state.show_summary_popup:
        st.write(st.session_state.summary)
        if st.button("Close Summary", key="close_summary_button"):
            st.session_state.show_summary_popup = False

    # Disable chat input if quiz, flashcards, or flashcard generation is active
    chat_input_disabled = (
        activity_active or
        "vectorstore" not in st.session_state or
        st.session_state.generating_flashcards or
        "flashcards" in st.session_state or
        "quiz_questions" in st.session_state
    )

    # Show "Clear Chat" button only if chat exists and no active activity
    if not activity_active and len(st.session_state.messages) > 0:
        spacer, clear_col = st.columns([4, 1])
        with clear_col:
            if st.button("🗑️ Clear Chat", key="clear_chat_button"):
                st.session_state.messages = []
                st.session_state.chat_started = False
                st.rerun()

    # Display moderation warning if needed
    if "moderation_warning" in st.session_state and st.session_state.moderation_warning:
        st.warning("Your input was flagged as inappropriate. Please try again.")

    display_chat_history()

    # Chat input is disabled when necessary
    if user_input := st.chat_input("Ask a question about your documents...", disabled=chat_input_disabled):
        st.session_state.chat_started = True
        handle_user_input(user_input, model)
        st.rerun()

if __name__ == "__main__":
    main()
