import streamlit as st
from tools import generate_summary, generate_flashcards, generate_quiz_questions
from utils import moderate_text, get_pdf_text, get_vectorstore
from langchain.chains import RetrievalQA

def process_uploaded_pdfs(pdf_docs, text_splitter):
    """Handles PDF upload, processing, and vector storage."""
    if not pdf_docs:
        return  # Do nothing if no PDFs uploaded

    process_button_disabled = not pdf_docs  # Disable processing if no PDFs
    if st.button("Process", disabled=process_button_disabled):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = text_splitter.split_text(raw_text)  # Use passed splitter
            vectorstore = get_vectorstore(text_chunks)
            
            # Store results in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.text_chunks = text_chunks  
            
            st.success("Documents processed!")

def generate_and_store_summary(text_chunks, model):
    """Generates and stores the summary with error handling in one function."""
    try:
        # Step 1: Check if text_chunks is available
        if not text_chunks:
            raise ValueError("No text chunks available for summarization.")
        
        # Step 2: Generate summary using the tool
        summary = generate_summary.invoke({"text_chunks": text_chunks, "model": model})

        # Step 3: Store the summary in session state
        st.session_state.summary = summary
        st.success("Summary Generated!")  # Notify the user

    except ValueError as e:
        st.error(f"Error: {e}")  # Handle missing text_chunks error
    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")  # Handle other exceptions


def generate_and_store_flashcards(text_chunks, model):
    try:
        flashcards = []
        for chunk in text_chunks:
            new_flashcards = generate_flashcards.invoke({"text_chunks": [chunk], "model": model})
            flashcards.extend(new_flashcards)

        st.session_state.flashcards = flashcards
        st.success(f"✅ Flashcards Generated: {len(flashcards)}")
    except Exception as e:
        st.error(f"❌ An error occurred during flashcard generation: {e}")



def generate_and_store_quiz(text_chunks, model):
    try:
        quiz_questions = generate_quiz_questions.invoke({"text_chunks": text_chunks, "model": model})

        # Store all types of questions
        st.session_state.quiz_questions = quiz_questions  
        st.session_state.current_question = 0  # Initialize question index
        st.session_state.score = 0  # Initialize score
        st.session_state.submitted = False  # Track question submission

        st.success(f"{len(quiz_questions)} Quiz questions generated!")
    except Exception as e:
        st.error(f"An error occurred during quiz question generation: {e}")


# Checker for flashcards and quiz, for the user to not be able to chat with the bot while taking the flashcards and quiz
def is_activity_active():
    """Check if quiz or flashcards are active to disable chat input."""
    quiz_active = (
        "quiz_questions" in st.session_state 
        and st.session_state.quiz_questions 
        and st.session_state.get("current_question", 0) < len(st.session_state.quiz_questions)
    )
    
    flashcard_active = (
        "flashcards" in st.session_state 
        and st.session_state.flashcards 
        and st.session_state.get("current_flashcard", 0) < len(st.session_state.flashcards)
    )
    
    return quiz_active or flashcard_active

def initialize_chat_history():
    """Initialize chat history if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False

def display_chat_history():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(user_input, model):
    """Process user input and generate a response using LLM."""
    if not user_input:
        return

    if not moderate_text(user_input):  # Check if input is safe
        st.warning("Your input was flagged as inappropriate. Please try again.")
        st.session_state.moderation_warning = True #set warning flag
        return

    # Reset the warning flag if the moderation passes
    if "moderation_warning" in st.session_state:
        st.session_state.moderation_warning = False

    # Append user input to session messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    vectorstore = st.session_state.vectorstore
    qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=vectorstore.as_retriever())

    with st.spinner("Generating Answer..."):
        try:
            answer_dict = qa_chain.invoke(user_input)
            answer = answer_dict['result']

            # Append LLM response
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")  # Handle LLM errors
