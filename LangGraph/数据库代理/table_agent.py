from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from typing import List

from my_tools import company_query


class TableModel(BaseModel):
    """该数据库的表的名称"""
    table_names: List[str] = Field(description="表的名字")


_table_system_template = """
你是一位注重细节的SQL专家。
给定一个查询目标，输出一个语法正确的SQLite查询语句，然后调用工具查询的结果并返回答案。

工具：
{tools_name}

输出：
{output_format}
"""
# output：
# {output_format}
_table_human_template = """
查询目标：获得所有的表名，不要有表名以sqlite开头的表。
"""

class Table:

    def __init__(self, llm):

        _tools = [company_query]

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _table_system_template),
            ("human",_table_human_template)
        ])
        _prompt = _prompt.partial(tools_name = ",".join([_tool.name for _tool in _tools]))
        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)

        self._parser = JsonOutputParser(pydantic_object=TableModel)
        # self._parser = StrOutputParser()
        _prompt = _prompt.partial(
            output_format=self._parser.get_format_instructions())

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

    _table = Table(_llm)
    _rt = _table({})

    print(_rt)
