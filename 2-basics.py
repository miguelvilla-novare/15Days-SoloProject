from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from commons import init_model

model = init_model()

messages = [
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
]

prompt_template = ChatPromptTemplate(messages=messages)

chain = prompt_template | model | StrOutputParser()

print(chain.invoke({"question": "What is a solar eclipse?"}))
