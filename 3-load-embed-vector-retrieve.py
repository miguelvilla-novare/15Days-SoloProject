from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from commons import init_model, init_embedding

model = init_model()
embedding = init_embedding()

path = "resources/"
directory_loader = PyPDFDirectoryLoader(path)
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini")
docs = directory_loader.load_and_split(text_splitter=text_splitter)
vector_store = FAISS.from_documents(documents=docs, embedding=embedding)

retriever = vector_store.as_retriever()

results: list[Document] = retriever.invoke("Evaluation Prompts")

for document in results:
    print(document.page_content)
