from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser #输出的JSON解析器

class Outline(BaseModel):
    main_title:str=Field(description="一级目录")
    sub_title:list[str]=Field(description="二级目录")

class Scheme(BaseModel):
    abstract:str=Field(description="全文摘要")
    outlines:list[Outline]=Field(description="项目大纲")
#定义提示语模板，越详细越好
_planner_system_template = r"""
您是一个优秀的方案规划师。
您的职责是根据标题，规划方案大纲。

要求：
1.大纲规划的层级中不需要序号。
2.另外还需要编写一个全文摘要以便更好的为后续完成该方案服务。

输出：
{output_format}
"""

_planner_human_template = r"""
题目：{title}
"""

class Planner:
    def __init__(self,llm):
        self._llm = llm
        # 设置提示语（设置岗位）
        _promot=ChatPromptTemplate([
            ("system",_planner_system_template),
            ("human",_planner_human_template)
        ])
        _parser=JsonOutputParser(pydantic_object=Scheme) #解析输出的json文件
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

    planner=Planner(_llm)
    answer=planner({"title":"基于大模型驱动的数字人"}) #输出为JSON格式
    print(answer["abstract"])
    
    
