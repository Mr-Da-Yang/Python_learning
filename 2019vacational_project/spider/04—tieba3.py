import requests


class TiebaSpider:
    def __init__(self, tieba_name):
        self.name = tieba_name
        self.url_temp = "https://tieba.baidu.com/f?kw=" + tieba_name + "&ie=utf-8&pn={}"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400'}

    def get_url(self):
        return [self.url_temp.format(i * 50) for i in range(2)]

    def parse_url(self, url):
        response = requests.get(url, headers=self.headers)
        return response.content.decode()

    def save(self, html_str, page):
        file_path = "{}--第{}--页.html".format(self.name, page)
        with open(file_path, 'w', encoding='utf-8')as f:
            f.write(html_str)

    def run(self):
        # 1列表
        self.get_url()
        # 2请求
        for url in self.get_url():
            html_str = self.parse_url(url)
            # 保存
            page = self.get_url().index(url) + 1
            self.save(html_str, page)


if __name__ == '__main__':
    tieba_spider = TiebaSpider('李毅')
    tieba_spider.run()
