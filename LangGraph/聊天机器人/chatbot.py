from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Annotated
from utils_tool import websearch,rag
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import uuid  # 生成唯一的标识符
# 创建工具


@tool
def chatbot_search(query: Annotated[str, "需要查询的问题的描述"]):
    """
    可以使用chatbot来搜索互联网信息
    """
    print("websearch工具搜索中...")
    return websearch(query)

@tool
def RAG_search(query:Annotated[str,"需要查询的人物的信息"]):
    """如果用户查询的信息是关于人物的信息，使用chatbot_retrieve工具"""
    print("RAG工具运行中...")
    return rag(query)
class Chatbot:
    def __init__(self):
        # 构建工具
        self._tools = [chatbot_search,RAG_search]
        # 导入模型
        self._llm = ChatOpenAI(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://192.168.10.11:60006/v1",
            temperature=0.7
        ).bind_tools(self._tools)

        self._graph = self._init_graph()
    # 构建图

    def _init_graph(self):
        _builder = StateGraph(MessagesState)
        _builder.add_node("chat_node", self._chat_agent)
        _builder.add_node("tool_node",ToolNode(tools=self._tools))

        _builder.add_edge(START, "chat_node")
        _builder.add_conditional_edges("chat_node",self._choose_tool)
        _builder.add_edge("tool_node", "chat_node")
        # 添加记忆，要增添configbye
        _memory = MemorySaver()
        return _builder.compile(checkpointer=_memory)
    # 创建聊天结点

    def _chat_agent(self, state):
        _message = self._llm.invoke(state["messages"])
        return {"messages": [_message]}

    def _choose_tool(self, state):
        _last_message = state["messages"][-1]

        if _last_message.tool_calls:
            return "tool_node"
        else:
            return END

    def __call__(self):
        # uuid4版本基于随机数生成，提供较高的随机性，是最常用的版本之一
        _thread_id = uuid.uuid4()
        while True:
            _human = input("user:")
            if _human == "bye":
                print("bye")
                break
            _config = {"thread_id": _thread_id}
            _ai = self._graph.invoke(
                {"messages": [HumanMessage(_human)]}, config=_config)
            # print(_ai)
            print(f"AI:{_ai['messages'][-1].content}")


if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot()
