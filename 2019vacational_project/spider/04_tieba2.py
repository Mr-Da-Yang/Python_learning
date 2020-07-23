import requests


class TiebaSpider:
    def __init__(self, tieba_name):
        self.tiebaname = tieba_name
        self.url = "https://tieba.baidu.com/f?kw=" + tieba_name + "&ie=utf-8&pn={}"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400'}

    def get_url(self):
        get_url = [self.url.format(i * 50) for i in range(2)]
        return get_url

    def save(self):
        file_path = "{}_第{}页000.html"
        with open(file_path.format(self.tiebaname, self.page), 'w', encoding='utf-8') as file:
            file.write(self.response.content.decode())

    def run(self):
        # 1获取url列表
        self.get_url()
        # 2发送请求
        for url in self.get_url():
            self.response = requests.get(url, headers=self.headers)
            self.page = self.get_url().index(url) + 1
            print(self.response.url)
            # 3保存
            self.save()


if __name__ == '__main__':
    tieba = TiebaSpider("李毅")
    tieba.run()
