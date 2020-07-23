import requests
import json


class Fanyi:
    def __init__(self, query_string):
        self.url = "https://fanyi.baidu.com/basetrans"
        self.query_string = query_string
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone;CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML,like Gecko) Version/11.0 Mobile/15A372 Safari/604.1'}
        self.langdetect = 'http://fanyi.baidu.com/langdetect'
    def get_post_data(self):
        #自动识别中英文
        # data = {"query": self.query_string}
        # json_str = self.parse_url(self.langdetect, data)
        # lan = json.loads(json_str)['lan']
        # to = 'en' if lan == "zh" else 'zh'
        post_data = {"query": self.query_string,
                     "from": lan,
                     "to": to}

        return post_data

    def parse_url(self, url, data):
        response = requests.post(url, data=data, headers=self.headers)

        return response.content.decode()

    def get_ret(self, json_str):
        temp_dict = json.loads(json_str)
        ret = temp_dict["trans"][0]["dst"]
        print("{}的翻译结果是:{}".format(self.query_string, ret))

    def run(self):
        post_data = self.get_post_data()
        json_str = self.parse_url(self.url, post_data)
        self.get_ret(json_str)


if __name__ == '__main__':
    fanyi = Fanyi("我们")
    fanyi.run()
