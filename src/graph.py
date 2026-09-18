from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.tools import RETRIEVAL_TOOLS

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def build_rag_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(RETRIEVAL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        system_prompt = SystemMessage(
            content="You are an enterprise technical assistant. Use the internal documentation tool "
                    "to answer questions accurately. If you don't find information, state that clearly."
        )
        all_messages = [system_prompt] + list(messages)
        response = llm_with_tools.invoke(all_messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "retrieve"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("retrieve", ToolNode(RETRIEVAL_TOOLS))

    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {
        "retrieve": "retrieve",
        END: END
    })
    builder.add_edge("retrieve", "agent")

    return builder.compile()

rag_app = None

def get_rag_app():
    global rag_app
    if rag_app is None:
        rag_app = build_rag_graph()
    return rag_app
