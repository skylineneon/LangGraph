from writer import Writer
from planner import Planner
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.types import Send
from typing import Annotated
from operator import add
#构建状态


# 大纲的状态
class Outline(TypedDict):
    title:str
    sub_title:list[str]    
#方案的状态
class SchemeState(TypedDict):
    title:str
    abstract:str
    outlines:list[Outline] #大纲
    content:Annotated[list[str],add]
    article:str

class ContentState(TypedDict):
    title: str
    abstract: str
    main_title:str
    sub_title:str

# 

class Designer:
    def __init__(self,llm):
        # 初始化模型
        self.llm = llm
        self._planner=Planner(self.llm)
        self._writer=Writer(self.llm)
        self._graph=self._init_graph()
    # 构建图
    def _init_graph(self):
        _budier=StateGraph(SchemeState)
        
        _budier.add_node("_planner_node",self._planner_node)
        _budier.add_node("_writer_node",self._write_node)
        _budier.add_node("_gather_node",self._gather_node)

        _budier.add_edge(START,"_planner_node")
        _budier.add_conditional_edges("_planner_node",self._dispaatch_edge)
        _budier.add_edge("_writer_node","_gather_node")
        _budier.add_edge("_gather_node",END)

        _graph=_budier.compile()
        return _graph
    # 构建规划者节点
    def _planner_node(self,state):
        while True:
            try:
                return self._planner(state)
            except Exception as e:
                print(e)
        
    # 构建写手节点
    def _write_node(self,state:ContentState):
        while True:
            try:
                _rt=self._writer(state) 
                return {"content":[_rt["content"]]}
            except Exception as e:
                print(e)   
    # 组合节点
    def _gather_node(self,state):
        _article = f"{state['title']}\n\n\n"
        _k = 0
        for _i,_outline in enumerate(state["outlines"]):
            _article+=f"{_i+1}.{_outline['main_title']}\n\n"
            for _j,_sub_title in enumerate(_outline["sub_title"]):
                _article+=f"{_i+1}.{_j+1}.{_sub_title}\n\n"
                _article+=f"{state['content'][_k]}\n\n\n\n"
                _k+=1

        return {"article":_article}

    # 定义一个条件
    def _dispaatch_edge(self,state):
        _title = state["title"]
        _abstract = state["abstract"]
        _outlines = state["outlines"]
        ret = []
        for _outline in _outlines:
            main_title = _outline["main_title"]
            for _sub_title in _outline["sub_title"]:
                ret.append(Send("_writer_node", {"title": _title,
                                                 "abstract": _abstract,
                                                 "main_title": main_title,
                                                 "sub_title": _sub_title
                                                 }))
        return ret

    def __call__(self,title):
        return self._graph.invoke({"title":title})


if __name__=="__main__":
    from langchain_openai import ChatOpenAI

    llm=ChatOpenAI(
        api_key="ollam",
        model_name="qwen2.5:7b",
        base_url="http://192.168.10.11:60006/v1"
    )
    design=Designer(llm)
    rt = design("基于大模型驱动的数字人")
    # print(rt["article"])
    with open("方案编写/design.txt","w+") as f:
        f.write(rt["article"])



