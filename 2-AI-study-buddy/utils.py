# utils.py
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from streamlit import json
from commons import init_model, init_embedding, init_moderation
from langchain.chains import LLMChain
from openai import OpenAI
import re


embedding = init_embedding()  # Initialize the embedding from commons.py
moderation_client = init_moderation()


def get_vectorstore(text_chunks):
    # Use the initialized embedding from commons.py
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore

def moderate_text(text: str) -> bool:
    try:
        # Create a moderation request (model is already set in init_moderation)
        response = moderation_client.moderations.create(
            input=text, 
            model="omni-moderation-latest"
        )    
        is_flagged = response.results[0].flagged

        if is_flagged:
            return False # Text is unsafe
        else: 
            return True # Text is safe
        
    except Exception as e:
        print(f"Error during moderation: {e}")
        return False  # Assume unsafe in case of error
    

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def parse_questions(questions_text):
    """Parses the generated questions text into a list of dictionaries."""
    questions = []
    if not questions_text:  # Handle empty or None input
        return questions

    question_blocks = questions_text.strip().split("\n\n")

    for block in question_blocks:
        try:
            question_data = {}
            lines = block.strip().split("\n")
            for line in lines:
                key, value = line.split(":", 1)
                question_data[key.strip()] = value.strip()

            # Ensure all required keys are present (important for robust parsing)
            required_keys = ["Question", "Type", "Options", "Correct Answer", "Explanation"]
            if all(key in question_data for key in required_keys):
                questions.append(question_data)
            else:
                print(f"Incomplete question data: {question_data}")  # Print to debug
        except Exception as e:
            print(f"Error parsing question block: {e}")  # Print to debug

    return questions

def parse_flashcards(text: str) -> list:
    """Parses AI-generated flashcards into a structured list."""
    flashcards = []
    flashcard_blocks = text.strip().split("\n\n")  # Split by blank lines

    for block in flashcard_blocks:
        if "Front:" in block and "Back:" in block:
            front = block.split("Front:")[1].split("Back:")[0].strip()
            back = block.split("Back:")[1].strip()
            flashcards.append({"front": front, "back": back})

    return flashcards
