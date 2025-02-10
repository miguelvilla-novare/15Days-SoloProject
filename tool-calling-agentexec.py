from langchain_core.tools import tool
from commons import init_model
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents import AgentType, initialize_agent

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

tools = [add, multiply]

llm = init_model()  # Initialize your LLM (make sure it's an LLM object, e.g., OpenAI())

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Optional: for debugging
)

query = input("Ask your questions: ")
response = agent.run(query) # Use agent.run
print(response)