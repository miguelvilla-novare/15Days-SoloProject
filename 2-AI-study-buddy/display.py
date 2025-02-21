import streamlit as st

def display_quiz():
    if "quiz_questions" in st.session_state and st.session_state.quiz_questions:
        if "current_question" not in st.session_state:
            st.session_state.current_question = 0
        if "score" not in st.session_state:
            st.session_state.score = 0
        if "submitted" not in st.session_state:
            st.session_state.submitted = False

        if st.session_state.current_question < len(st.session_state.quiz_questions):
            question_data = st.session_state.quiz_questions[st.session_state.current_question]

            st.subheader(f"Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_questions)}")
            st.markdown(f"**{question_data.get('Question', 'Question text not available')}**")

            question_type = question_data.get("Type")

            # Multiple-Choice Question Handling
            if question_type == "multiple_choice":
                options_str = question_data.get("Options", "")
                options_list = options_str.split(",")

                options = {}
                option_letters = ['A', 'B', 'C', 'D']

                for i, opt in enumerate(options_list):
                    if i < len(option_letters):
                        letter = option_letters[i]
                        option_text = opt.split(")")[1].strip() if ")" in opt else opt.strip()
                        options[f"{letter}) {option_text}"] = letter  # Store only letter for answer checking

                if options:
                    user_answer_text = st.radio("Select an answer", list(options.keys()), disabled=st.session_state.submitted)
                    user_answer_letter = options.get(user_answer_text)

            # True/False Question Handling
            elif question_type == "true_false":
                options = {"A) True": "A", "B) False": "B"}  # Fixed True/False options
                user_answer_text = st.radio("Select an answer", list(options.keys()), disabled=st.session_state.submitted)
                user_answer_letter = options.get(user_answer_text)

            else:
                st.error("Invalid question type!")
                user_answer_letter = None

            # Submission Logic - Ensures the score is updated only once per question
            if not st.session_state.submitted:
                if st.button("Submit"):
                    st.session_state.submitted = True
                    correct_answer = question_data.get("Correct Answer")
                    explanation = question_data.get("Explanation", "No explanation available.")

                    # Store the correct answer and explanation for display
                    st.session_state.correct_answer = correct_answer
                    st.session_state.explanation = explanation

                    # Only increment score if the answer is correct and has not been counted yet
                    if user_answer_letter == correct_answer:
                        st.session_state.score += 1

                    st.rerun()  # Refresh the UI immediately after submission

            # Show Answer After Submission
            else:
                correct_answer = st.session_state.correct_answer
                explanation = st.session_state.explanation

                if user_answer_letter == correct_answer:
                    st.success(f"Your answer is correct! 🎉\n\n**Explanation:** {explanation}")
                else:
                    st.error(f"❌ Incorrect. The correct answer is **{correct_answer}**.\n\n**Explanation:** {explanation}")

                # Restrict user to click only "Next" after submission
                if st.button("Next"):
                    st.session_state.current_question += 1
                    st.session_state.submitted = False  # Reset submission state for the next question
                    st.rerun()

        else:
            st.subheader("🎉 Quiz Completed!")
            st.write(f"Your final score: **{st.session_state.score} / {len(st.session_state.quiz_questions)}**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Restart Quiz"):
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.submitted = False
                    st.rerun()

            with col2:
                if st.button("End Quiz"):
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.submitted = False
                    st.session_state.quiz_questions = []
                    st.rerun()



def display_flashcards():
    """Handles the display and interaction of flashcards."""
    if "flashcards" in st.session_state and st.session_state.flashcards:
        if "current_flashcard" not in st.session_state:
            st.session_state.current_flashcard = 0
            st.session_state.reveal_answer = False  # Track answer visibility

        flashcards = st.session_state.flashcards
        total_flashcards = len(flashcards)

        st.markdown(f"<h3 style='text-align: center;'>🃏 Flashcard {st.session_state.current_flashcard + 1} of {total_flashcards}</h3>", unsafe_allow_html=True)

        flashcard = flashcards[st.session_state.current_flashcard]
        st.markdown(f"<h4 style='text-align: center;'> {flashcard['front']}</h4>", unsafe_allow_html=True)

        # Reveal Answer Button - Centered
        col = st.columns([2, 1, 2])
        with col[1]:
            if not st.session_state.reveal_answer:
                if st.button("💡 Reveal Answer", key="reveal_button"):
                    st.session_state.reveal_answer = True
                    st.rerun()

        # Show Answer if Revealed
        if st.session_state.reveal_answer:
            st.markdown(f"<h4 style='text-align: center;'> {flashcard['back']}</h4>", unsafe_allow_html=True)

        st.markdown("---")

        # Navigation Buttons - Adjusting column widths for better alignment
        col1, col2, col3 = st.columns([1, 2, 1])  # Balanced spacing

        with col1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if st.button("⬅️ Previous", disabled=st.session_state.current_flashcard == 0):
                st.session_state.current_flashcard -= 1
                st.session_state.reveal_answer = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if st.button("❌ Exit Flashcards"):
                for key in ["current_flashcard", "reveal_answer", "flashcards"]:
                    st.session_state.pop(key, None)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if st.session_state.current_flashcard < total_flashcards - 1:
                if st.button("➡️ Next"):
                    st.session_state.current_flashcard += 1
                    st.session_state.reveal_answer = False
                    st.rerun()
            else:
                if st.button("🏁 Exit Flashcards"):
                    for key in ["current_flashcard", "reveal_answer", "flashcards"]:
                        st.session_state.pop(key, None)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)