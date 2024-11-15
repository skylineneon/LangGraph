from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel,Field

class SubScheme(BaseModel):
    content:str=Field(description="生成的内容")
# 可以用大模型润色一下
_writer_system_template = r"""
您是一个优秀的方案写手。
您的职责在于，基于项目名称、全文摘要以及所提供的一级和二级目录结构，精心撰写相应章节的内容。您需要确保每个目录下的文字详实、准确，与项目的主题和摘要精神保持一致，同时确保内容的连贯性和逻辑性。

要求：
1.字数越多越好。
2.不要有序号。

输出：
{output_format}
"""

_writer_human_template = r"""
项目标题：{title}
全文摘要：{abstract}
一级目录：{main_title}
二级目录：{sub_title}
"""
class Writer:
    def __init__(self,llm):
        self._llm = llm
        _promot=ChatPromptTemplate([
            ("system",_writer_system_template),
            ("human",_writer_human_template)
        ])
        _parser=JsonOutputParser(pydantic_object=SubScheme)
        _promot=_promot.partial(output_format=_parser.get_format_instructions())

        self._chain=_promot|self._llm|_parser
    def __call__(self,state):
        return self._chain.invoke(state)
if __name__=="__main__":
    from langchain_openai import ChatOpenAI

    _llm=ChatOpenAI(
        api_key="ollam",
        model_name="qwen2.5:7b",
        base_url="http://192.168.10.11:60006/v1"
    )
    writer=Writer(_llm)
    answer=writer({"title":"基于大模型驱动的数字人",
                   "abstract":"本方案旨在构建一个基于大规模预训练语言模型的数字人系统。该系统将通过深度学习技术，实现高度智能化和个性化的交互体验。具体而言，我们将详细规划从数据准备、模型选择与训练、对话逻辑设计到最终部署及优化的全过程。",
                   "main_title":"数字人系统",
                   "sub_title":"团队结构与角色划分"
                   })
    print(answer["content"])
    # print(answer)