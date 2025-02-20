
from pydantic import BaseModel, Field
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from utils import parse_questions, parse_flashcards

class GetSummarySchema(BaseModel):
    text_chunks: list[str] = Field(..., description="List of text chunks to summarize")

# Summarization function (takes `model` as a parameter)
@tool
def generate_summary(text_chunks: list[str], model) -> str:
    """Generates a summary from the given text chunks using the provided model."""
    
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Analyze the provided text and extract the key information concisely. If the text is structured into sections, "
            "identify and summarize each section separately. If the text is unstructured, generate a well-organized summary "
            "covering the main points. Keep the summaries clear, concise, and informative.\n\n"
            "**Text:**\n{text}\n\n"
            "**Summary:**\n[Generate a structured or free-form summary based on the document format.]"
        )
    )

    chain_type = "map_reduce" if len(docs) > 5 else "stuff"
    chain = load_summarize_chain(model, chain_type=chain_type, prompt=summary_prompt)

    summary_dict = chain.invoke(docs)
    return summary_dict["output_text"]

@tool
def generate_quiz_questions(text_chunks: list, model) -> list:
    """Generates a mix of multiple-choice and true/false quiz questions from a list of text chunks."""

    question_prompt = PromptTemplate(
        input_variables=["text"],
        template="Generate a mix of MULTIPLE-CHOICE and TRUE/FALSE quiz questions based on the following text:\n{text}.\n\n"
                 "The number of questions should be proportional to the length of the text.\n\n"
                 "Each question MUST be formatted like this:\n"
                 "Question: The question text\n"
                 "Type: multiple_choice or true_false\n"
                 "Options: (For multiple-choice) A) Option A, B) Option B, C) Option C, D) Option D\n"
                 "Options: (For true/false) A) True, B) False\n"
                 "Correct Answer: A or B or C or D (for multiple-choice), A or B (for true/false)\n"
                 "Explanation: Explanation of the answer\n\n"
                 "Separate each question with a blank line.\n\n"
                 "Return ONLY the questions in the specified format. Do not include any other text."
    )

    all_questions = []
    for chunk in text_chunks:
        chain = question_prompt | model  # Create the RunnableSequence
        response = chain.invoke({"text": chunk})  # Invoke the chain, passing input as a dictionary
        questions_str = response.content  # Access the content (string)

        try:
            questions_list = parse_questions(questions_str)
            if isinstance(questions_list, list) and len(questions_list) > 0:
                all_questions.extend(questions_list)
            else:
                print(f"LLM did not return questions in the correct format. Raw string: {questions_str}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Raw string: {questions_str}")

    return all_questions

@tool
def generate_flashcards(text_chunks: list, model) -> list:
    """Generates flashcards from text chunks where the front is a question and the back is an answer."""

    flashcard_prompt = PromptTemplate(
        input_variables=["text"],
        template="Generate concise and useful flashcards from the following text:\n{text}\n\n"
                 "Each flashcard should have:\n"
                 "- **Front (Question):** A key concept in question form.\n"
                 "- **Back (Answer):** The answer or explanation.\n\n"
                 "Format each flashcard like this:\n"
                 "Front: [Question]\n"
                 "Back: [Answer]\n\n"
                 "Return ONLY the flashcards in this format, without extra text."
    )

    all_flashcards = []
    for chunk in text_chunks:
        chain = flashcard_prompt | model
        response = chain.invoke({"text": chunk})

        try:
            # Extract flashcards
            flashcards_list = parse_flashcards(response.content)
            if isinstance(flashcards_list, list) and flashcards_list:
                all_flashcards.extend(flashcards_list)
            else:
                print(f"LLM did not return flashcards correctly. Raw response: {response.content}")
        except Exception as e:
            print(f"Error parsing flashcards: {e}. Raw response: {response.content}")

    return all_flashcards



