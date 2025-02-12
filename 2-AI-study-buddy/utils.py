# utils.py
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from commons import init_model, init_embedding
from langchain.chains import LLMChain

model = init_model()  # Initialize the model from commons.py
embedding = init_embedding()  # Initialize the embedding from commons.py
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini")


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

    summary = chain.run(docs)  # Pass the list of Documents
    return summary


def generate_questions(text_chunks, model):
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template="Generate EXACTLY 5 diverse and insightful questions based on the following text:\n{text}. Generate the question as if you can also answer that question. List each question on a new line."
    )
    chain = LLMChain(llm=model, prompt=question_prompt)

    # Combine ALL text chunks into a SINGLE string for question generation
    combined_text = " ".join(text_chunks)  # Join the chunks

    questions_str = chain.run(combined_text) # Generate questions from combined text
    questions = questions_str.splitlines()
    cleaned_questions = [q.strip() for q in questions if q.strip()]

    # Truncate or pad as needed (important!)
    while len(cleaned_questions) > 5:
        cleaned_questions.pop()

    while len(cleaned_questions) < 5:
        cleaned_questions.append("Question not generated") # Or some other placeholder

    return cleaned_questions