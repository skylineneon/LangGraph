from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# 定义模版


class Plan(BaseModel):
    """制定的计划"""
    task_list: List[str] = Field(description="子任务列表")


_planner_system_template = """
您具备优秀的任务规划能力，将精心构思一套任务解决的步骤，以应对用户提出的问题。不要过于复杂。

输出：
{output_format}
"""

_planner_human_template = """
查询目标：
{query}
"""


class Planner:

    def __init__(self, llm):

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _planner_system_template),
            ("human", _planner_human_template)
        ])
        # 解析JSON格式
        _parser = JsonOutputParser(pydantic_object=Plan)
        # 将解析到的格式加入到_prompt中
        _prompt = _prompt.partial(
            output_format=_parser.get_format_instructions())

        self._chain = _prompt | llm | _parser

    def __call__(self, state):
        return self._chain.invoke(state)


if __name__ == '__main__':

    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="http://58.34.49.226:11046/v1",
        model="qwen2.5:14b",
        api_key="ollama"
    )

    _planner = Planner(_llm)
    _rt = _planner({"query": "2024年法国巴黎奥运会女子10米跳水冠军的父亲是谁"})

    print(_rt)
