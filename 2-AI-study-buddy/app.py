import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA  # For question answering
from commons import init_model, init_moderation
from langchain_text_splitters import TokenTextSplitter
from utils import get_pdf_text, get_vectorstore, get_summary, generate_quiz_questions, moderate_text  # Import the quiz function

model = init_model()  # Initialize the model from commons.py
moderation = init_moderation()
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
                st.session_state.text_chunks = text_chunks  # Store text chunks
                
        st.subheader("Actions")  # Subheader for clarity

        # Button container for Generate buttons
        st.container()
        col1, col2 = st.columns(2)  # 2 columns for Generate buttons

        get_summary_button_disabled = "text_chunks" not in st.session_state  # Disable if no processing
        if col1.button("Generate Summary", key="get_summary_button", disabled=get_summary_button_disabled) and "text_chunks" in st.session_state:  # New button and check
            with st.spinner("Generating Summary..."):
                try:
                    summary = get_summary(st.session_state.text_chunks, model)  # Use stored text_chunks
                    st.session_state.summary = summary
                    st.success("Summary Generated!")
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")

        show_summary_button_disabled = "summary" not in st.session_state
        if col2.button("Show Summary", key="show_summary_button", disabled=show_summary_button_disabled) and "summary" in st.session_state:
            st.session_state.show_summary_popup = True
    

        # Full-width buttons below
        if col1.button("Generate Flashcards", disabled="text_chunks" not in st.session_state):
            with st.spinner("Generating Flashcards..."):
                try:
                    flashcards = get_flashcards(st.session_state.text_chunks, model)  # Use stored text_chunks
                    st.session_state.flashcards = flashcards
                    st.success("Flashcards Generated!")
                except Exception as e:
                    st.error(f"An error occurred during flashcard generation: {e}")

        if col2.button("Take Quiz", disabled="text_chunks" not in st.session_state):
            with st.spinner("Generating Quiz..."):
                try:
                    quiz_questions = generate_quiz_questions(st.session_state.text_chunks, model)
                    # Filter out non-multiple choice questions
                    st.session_state.quiz_questions = [q for q in quiz_questions if q.get("Type") == "multiple_choice"]
                    st.session_state.current_question = 0  # Initialize question index
                    st.session_state.score = 0  # Initialize score
                    st.session_state.submitted = False  # Track question submission
                    st.success("Quiz questions generated!")
                except Exception as e:
                    st.error(f"An error occurred during quiz question generation: {e}")
                    
        

    if "quiz_questions" in st.session_state and st.session_state.quiz_questions:
        if st.session_state.current_question < len(st.session_state.quiz_questions):
            question_data = st.session_state.quiz_questions[st.session_state.current_question]

            st.subheader(f"Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_questions)}")
            st.write(question_data.get("Question", "Question text not available"))

            question_type = question_data.get("Type")

            if question_type == "multiple_choice":
                options_str = question_data.get("Options", "")
                options_list = options_str.split(",")
                options = {opt.split(")")[1].strip(): opt.split(")")[0].strip() for opt in options_list if ")" in opt}

                if options:
                    user_answer_text = st.radio("Select an answer", list(options.keys()), disabled=st.session_state.submitted)
                    user_answer_letter = options.get(user_answer_text)

                    if st.button("Submit") and not st.session_state.submitted:
                        st.session_state.submitted = True  # Mark question as submitted

                        # Check answer
                        if user_answer_letter == question_data.get("Correct Answer"):
                            explanation = question_data.get("Explanation", "No explanation available.")
                            st.success(f"""Your answer is correct!  
                                            **Explanation:** {explanation}""")
                            st.session_state.score += 1
                        else:
                            st.error(f"Incorrect. The correct answer is **{question_data.get('Correct Answer', 'Not available')}**.")

            # ✅ Add "Next" button after submission
            if st.session_state.submitted:
                if st.button("Next"):
                    st.session_state.current_question += 1  # Move to next question
                    st.session_state.submitted = False  # Reset submission state
                    st.rerun()  # Refresh to load next question

        else:
            # ✅ Display final score
            st.subheader("🎉 Quiz Completed!")
            st.write(f"Your final score: **{st.session_state.score} / {len(st.session_state.quiz_questions)}**")

            # ✅ Restart or End Quiz
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Restart Quiz"):
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.submitted = False
                    st.rerun()  # Restart quiz

            with col2:
                if st.button("End Quiz"):
                     # Reset quiz state, but keep processed_data
                    st.session_state.current_question = 0  # Reset, not delete
                    st.session_state.score = 0
                    st.session_state.submitted = False
                    st.session_state.quiz_questions = [] #Clear quiz questions
                    st.rerun()

    if "show_summary_popup" in st.session_state and st.session_state.show_summary_popup:  # Check the flag
        st.write(st.session_state.summary)

        if st.button("Close Summary", key="close_summary_button"):  # Add a close button
            st.session_state.show_summary_popup = False

        # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
        

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input section
    if user_input := st.chat_input("Ask a question about your documents..."):
        if user_input:  # Check if user_input is not empty
            if moderate_text(user_input):  # Check if input is SAFE
                # Input is safe, proceed with LLM processing
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
                    vectorstore = st.session_state.vectorstore
                    qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=vectorstore.as_retriever())

                    with st.spinner("Generating Answer..."):
                        try:  # Add try-except for LLM errors
                            answer_dict = qa_chain.invoke(user_input)
                            answer = answer_dict['result']

                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            with st.chat_message("assistant"):
                                st.markdown(answer)
                        except Exception as e:
                            st.error(f"Error generating answer: {e}") # Handle LLM errors

                else:
                    st.warning("Please process the documents first.")

            else:  # Input was flagged as inappropriate (moderate_text returned False)
                st.warning("Your input was flagged as inappropriate. Please try again.")

if __name__ == "__main__":
    main()