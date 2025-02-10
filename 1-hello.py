from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini",
                 base_url="https://ainovate.novare.com.hk")

print(llm.invoke("Hello world"))
