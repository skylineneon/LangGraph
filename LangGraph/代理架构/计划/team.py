from langchain_openai import ChatOpenAI
from planner import Planner
from executor import Executor
from gatherer import Gatherer
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Any, List
from typing_extensions import TypedDict
from operator import add
# 定义状态


class teamstate(TypedDict):
    query: str
    task_list: List[str]
    infos: Annotated[List[str], add]
    result: str


class Team:
    def __init__(self, llm):
        self.llm = llm
        self._planner = Planner(self.llm)
        self._executor = Executor(self.llm)
        self._gatherer = Gatherer(self.llm)
        self._graph = self._init_graph()
    # 构建图

    def _init_graph(self):
        _builder = StateGraph(teamstate)
        _builder.add_node("_planner_node", self._planner_node)
        _builder.add_node("_executor_node", self._executor_node)
        _builder.add_node("_gatherer_node", self._gatherer_node)

        _builder.add_edge(START, "_planner_node")
        _builder.add_edge("_planner_node", "_executor_node")
        _builder.add_conditional_edges("_executor_node", self._executor_router, {
                                       "_gatherer_node": "_gatherer_node", "_executor_node": "_executor_node"})
        _builder.add_edge("_gatherer_node", END)

        _graph = _builder.compile()
        return _graph
    # 构建节点

    def _planner_node(self, state):
        return self._planner(state)

    def _executor_node(self, state):
        _infos = state.get("infos", [])

        _task_index = len(_infos)
        _task = state["task_list"][_task_index]

        _rt = self._executor({
            "infos": _infos,
            "task": _task
        })

        return {"infos": [_rt]}

    def _gatherer_node(self, state):
        _infos = state["infos"]
        _query = state["query"]
        _rt = self._gatherer({
            "infos": _infos,
            "query": _query
        })

        return {"result": _rt}

    def _executor_router(self, state):
        _task_len = len(state["task_list"])
        _infos = len(state["infos"])
        if _task_len == _infos:
            return 
        else:
            return "_executor_node"

    def __call__(self, query):
        return self._graph.invoke({"query": query})


if __name__ == "__main__":
    llm = ChatOpenAI(
        base_url="http://58.34.49.226:11046/v1",
        model="qwen2.5:14b",
        api_key="ollama"
    )
    team = Team(llm)
    answer = team("2024年法国巴黎奥运会单人女子跳水冠军的家乡？")
    # print(answer["result"])
    print(answer)
