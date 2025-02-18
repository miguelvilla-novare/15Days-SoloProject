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


model = init_model()  # Initialize the model from commons.py
embedding = init_embedding()  # Initialize the embedding from commons.py
moderation_client = init_moderation()
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini")


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

def get_vectorstore(text_chunks):
    # Use the initialized embedding from commons.py
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore
    
def get_summary(text_chunks, model):
    # Create Document objects from the text chunks
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n{text}"
    )
    chain = load_summarize_chain(model, chain_type="stuff", prompt=summary_prompt)

    if len(docs) > 5:  # Check number of documents, not text_chunks
       chain = load_summarize_chain(model, chain_type="map_reduce", prompt=summary_prompt) #Consider refine as well

    summary = chain.invoke(docs)  # Pass the list of Documents
    return summary

def generate_quiz_questions(text_chunks, model):
    """Generates 5 multiple-choice quiz questions from text chunks (NO STREAMLIT DEPENDENCY)."""

    question_prompt = PromptTemplate(
        input_variables=["text"],
        template="Generate EXACTLY 5 diverse MULTIPLE-CHOICE quiz questions based on the following text:\n{text}.\n\n"
                 "Each question MUST be formatted like this:\n"
                 "Question: The question text\n"
                 "Type: multiple_choice\n"
                 "Options: A) Option A, B) Option B, C) Option C, D) Option D\n"
                 "Correct Answer: A or B or C or D\n"
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
            if isinstance(questions_list, list) and len(questions_list) == 5:
                all_questions.extend(questions_list)
            else:
                print(f"LLM did not return 5 questions or the format is incorrect. Raw string: {questions_str}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Raw string: {questions_str}")

    return all_questions[:5]


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