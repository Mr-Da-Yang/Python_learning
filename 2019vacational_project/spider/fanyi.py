import requests
import json
class Fanyi:
    def __init__(self, query_str):
        self.url = "https://fanyi.baidu.com/basetrans"
        self.query_str = query_str
        self.data = {"from": "zh",
                     "to": "en",
                     "query": query_str, }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/56.0.2924.75 Mobile/14E5239e Safari/602.1'}
    def get_response(self):

        response = requests.post(self.url, data=self.data, headers=self.headers)

        python_str = response.content.decode()
        a=json.loads(python_str)
        return a

    def display(self,query_str):

        display = query_str + "的翻译是："+self.get_response()['trans'][0]['dst']
        print(display)
    def run(self):
        #1.获取url

        #2.发送请求，获取响应
        self.get_response()
        #3.display
        self.display(self.query_str)


if __name__ == '__main__':
     fanyi = Fanyi("坚持就是胜利")
     fanyi.run()