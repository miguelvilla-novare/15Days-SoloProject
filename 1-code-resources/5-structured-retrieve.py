from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from pydantic import BaseModel, Field

from commons import init_model, init_embedding

model = init_model()
embedding = init_embedding()

path = "resources/"
directory_loader = PyPDFDirectoryLoader(path)
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini")
docs = directory_loader.load_and_split(text_splitter=text_splitter)
vector_store = FAISS.from_documents(documents=docs, embedding=embedding)

retriever = vector_store.as_retriever()


class StructuredAnswer(BaseModel):
    summary: str = Field(description="A concise summary of the answer.")
    key_points: List[str] = Field(..., description="A list of the main points.")
    follow_up_question: str = Field(description="A related question for further discussion.")


prompt = """
You are a knowledgeable assistant. Based on the provided context, answer the following question.

Context:
{context}

Question:
{question}
"""

structured_llm = model.with_structured_output(StructuredAnswer)


def retrieve_context(search_query):
    relevant_docs: list[Document] = retriever.invoke(search_query)
    return relevant_docs


def run_structured_qa(question):
    context = retrieve_context(question)
    filled_prompt = prompt.format(context=context, question=question)
    structured_output = structured_llm.invoke(filled_prompt)
    return structured_output


# Example usage
query = "Which evaluation prompts were used for Deepseek-R1?"
result = run_structured_qa(query)

print(f"Query: {query}")
print(f"Summary: {result.summary}\n")
print("Key Points: ")
for kp in result.key_points:
    print(f"- {kp}")
print("----------")
print(f"Consider asking: {result.follow_up_question}")
