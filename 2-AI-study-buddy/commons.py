from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
import os

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_NAME = "text-embedding-3-small"
BASE_URL = "https://ainovate.novare.com.hk/"
MODERATION_MODEL = "omni-moderation-latest"


load_dotenv()
def init_model() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL_NAME, base_url=BASE_URL)

def init_embedding() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_NAME, base_url=BASE_URL)

def init_moderation() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=BASE_URL)