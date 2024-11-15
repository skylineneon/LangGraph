from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from my_tools import company_query

_table_system_template = """
你是一位注重细节的SQL专家。
给定一个查询目标或执行目标，输出一个语法正确的SQLite查询或执行语句，然后调用工具执行它。
如果是查询，则返回查询结果。
如果是执行的添加，删除，更新数据，返回OK表示执行成功。
所有表的信息：
{infos}

工具：
{tools_name}

"""
# output：
# {output_format}
_table_human_template = """
查询目标：{query}
"""

class Exec:

    def __init__(self, llm):

        _tools = [company_query]

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _table_system_template),
            ("human",_table_human_template)
        ])
        _prompt = _prompt.partial(tools_name = ",".join([_tool.name for _tool in _tools]))
        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)

        self._parser = StrOutputParser()

        self._chain = _prompt | _llm_with_tools_agent
        

    def __call__(self, state):
        _rt = self._chain.invoke(state)
        _messages = _rt["messages"]
        return self._parser.invoke(_messages[-1])



if __name__ == '__main__':

    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="http://192.168.10.13:60001/v1",
        model="qwen2.5:7b",
        api_key="ollama"
    )

    _exec = Exec(_llm)
    _rt = _exec({"query":"统计下我们公司有多少男员工","infos":[]})

    print(_rt)
