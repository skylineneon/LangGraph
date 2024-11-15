from zhipuai import ZhipuAI


class WebSearch:

    def __init__(self):
        self._client = ZhipuAI(api_key="51109e5e1fd67c96a7a51eb74e5ae8ca.SCpBvbMmVT6i3axx")

        self._tools = [{
            "type": "web_search",
            "web_search": {
                "enable": True #默认为关闭状态（False） 禁用：False，启用：True。
            }
        }]
    
    def __call__(self,query):
        messages = [{
            "role": "user",
            "content": query
        }]

        response = self._client.chat.completions.create(
            model="glm-4-plus",
            messages=messages,
            tools=self._tools
        )
        return response.choices[0].message.content

websearch = WebSearch()


