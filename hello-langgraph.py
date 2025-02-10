from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from commons import init_model
from langgraph.constants import START, END

class State(TypedDict):

    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = init_model()

def invoke_chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


CHATBOT = "chatbot"

graph_builder.add_node(CHATBOT, invoke_chatbot)
graph_builder.add_edge(START, CHATBOT)
graph_builder.add_edge(CHATBOT, END)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break