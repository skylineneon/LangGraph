from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from utils import websearch
from typing import Annotated
from langgraph.prebuilt import create_react_agent


@tool
def web_search(query: Annotated[str, "互联网查询内容"]):
    """通过web_search工具查询互联网上的信息"""
    _rt = websearch(query)
    return _rt

_researcher_system_template = """
你是一个乐于助人的人工智能助手，与其他助手合作。
使用提供的工具来回答问题。
如果你不能完全回答，另一个助手用不同的工具。
这将有助于你取得进展。尽你所能取得进展。
您可以访问以下工具：
{tools_name}
"""



class Researcher:

    def __init__(self, llm):

        _tools = [web_search]

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _researcher_system_template),
            MessagesPlaceholder(variable_name="messages")
        ])

        _prompt = _prompt.partial(tools_name=",".join([_tool.name for _tool in _tools]))

        _llm_with_tools_agent = create_react_agent(llm,tools=_tools)
        
        self._chain = _prompt|_llm_with_tools_agent

    def __call__(self, state):
        _rt = self._chain.invoke(state)
        _messages = _rt["messages"]
        return _messages[-1]
    
if __name__ == '__main__':

    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="http://192.168.10.13:60001/v1",
        model="qwen2.5:7b",
        api_key="ollama"
    )

    _researcher = Researcher(_llm)
    _rt = _researcher({"messages":[("human","获取英国过去5年的国内生产总值。一旦你把它编码好，并执行画图，就完成。")]})

    print(_rt)