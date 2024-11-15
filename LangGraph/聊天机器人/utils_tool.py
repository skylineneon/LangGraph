from RAG_doc import _zhaoyuxuan, _sunyuexin
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from zhipuai import ZhipuAI




class WebSearch:
    def __init__(self):
        self._client = ZhipuAI(
            api_key="71dd6b1e11ccd6ba0e43bcd1daa98bac.gs2cKdpcBoqhi9Ly")
        self._tools = [{
            "type": "web_search",
            "web_search": {
                "enable": True
            }
        }]

    def __call__(self, query):
        messages = [{
            "role": "user",
            "content": query
        }]
        response = self._client.chat.completions.create(
            model="glm-4",
            messages=messages,
            tools=self._tools
        )
        return response.choices[0].message.content


websearch = WebSearch()


class RAG:
    def __init__(self):
        _docs = [
            Document(page_content=_zhaoyuxuan),
            Document(page_content=_sunyuexin)
        ]
        _emb = OllamaEmbeddings(
            base_url="http://192.168.10.11:60006",
            model="bge-m3"
        )

        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20)
        _splits = _text_splitter.split_documents(_docs)
        self._vectorstore = Chroma.from_documents(_splits, _emb)

    def __call__(self, query):
        return self._vectorstore.similarity_search_with_score(query=query, k=2)


rag = RAG()
