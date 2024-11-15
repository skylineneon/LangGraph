from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import Annotated
from utils import websearch
from langgraph.prebuilt import create_react_agent
# from langchain_community.utilities import SerpAPIWrapper
# import os
# os.environ["SERPAPI_API_KEY"] = (
#     "9dd2b2ee429ed996c75c1daf7412df16336axxxxxxxxxxxxxxx"
# )
# 定义工具


# @tool
# def serpapi_search(query: Annotated[str, "互联网查询标题"]):
#     """只有在需要了解实时信息 或 不知道的事情的时候 才会使用这个工具，需要传入要搜索的内容。"""
#     serp = SerpAPIWrapper()
#     print("使用工具")
#     result = serp.run(query)
#     return result


@tool
def web_search(query: Annotated[str, "互联网查询标题"]):
    """通过web_search工具查询互联网上的信息"""
    # print("调用了工具")
    _rt = websearch(query)
    return _rt


_executor_system_template = """
您是一个优秀的子任务执行者，您需要根据子任务的名称和参考信息，完成子任务的信息查询。
您可以使用以下工具来协助您更好的完成该任务：
{tools_name}
"""

_executor_human_template = """
参考信息：
{infos}
子任务的名称：
{task}
"""


class Executor:

    def __init__(self, llm):

        _tools = [web_search,]

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _executor_system_template),
            ("human", _executor_human_template)
        ])
        # 将工具名加入_prompt中
        _prompt = _prompt.partial(tools_name=",".join(
            [_tool.name for _tool in _tools]))
        # 创造一个带工具的Agent
        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)
        # 创建一个链
        # 链式调用通常要求前一个对象的输出类型与后一个对象的输入类型兼容,Agent返回的是Message,
        self._chain = _prompt | _llm_with_tools_agent
        self._parser = StrOutputParser()

    def __call__(self, state):
        _rt = self._chain.invoke(state)
        _messages = _rt["messages"]
        return self._parser.invoke(_messages[-1])


if __name__ == '__main__':

    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="http://58.34.49.226:11046/v1",
        model="qwen2.5:14b",
        api_key="ollama"
    )

    _executor = Executor(_llm)
    _rt = _executor({
        "infos": [],
        "task": "确定2024年法国巴黎奥运会女子10米跳水比赛的冠军"
    })

    print(_rt)
