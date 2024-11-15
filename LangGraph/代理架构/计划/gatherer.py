from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


_gatherer_system_template = """
您是一个优秀的助手，根据参考信息，准确的完成用户查询目标的任务。
只回答与提问相关的答案，不要引入其它信息。
"""

_gatherer_human_template = """
参考信息：
{infos}
查询目标：
{query}
"""


class Gatherer:

    def __init__(self, llm):

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _gatherer_system_template),
            ("human", _gatherer_human_template)
        ])

        _parser = StrOutputParser()
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

    _gatherer = Gatherer(_llm)
    _rt = _gatherer({
        "infos": ["全红禅是2024年奥运会跳水冠军冠军", "她住在广东"],
        "query": "2024年法国巴黎奥运会女子10米跳水冠军的家乡在哪里"
    })

    print(_rt)
