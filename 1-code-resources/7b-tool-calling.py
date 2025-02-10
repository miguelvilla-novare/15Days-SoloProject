from langchain_core.tools import tool
from commons import init_model
from langchain.schema import HumanMessage, AIMessage

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

tools = [add,multiply]
# Initialize the model and bind the tools
model = init_model()
model.bind_tools(tools)

query = input("Please enter your query: ")

messages = [HumanMessage(query)]

response = model.invoke(messages)

print(response.pretty_print())
messages.append(response)

for tool_call in response.tool_calls:
    selected_tool = {'add': add, 'multiply': multiply[tool_call]}
    tool_msg = selected_tool.invoke(tool_call)
    print(tool_msg.prettyprint())
    messages.append(tool_msg)