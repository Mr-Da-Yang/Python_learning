import requests
import re
import json
class Guokr:
    def __init__(self):
       self.url ='https://www.guokr.com/ask/highlight/?page={}'
       self.headers ={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}

    def get_url_list(self):
        return [self.url.format(i) for i in range(1,101)]

    def parse_url(self,url):
        html_str = requests.get(url,self.headers).content.decode()
        return html_str

    def extract_content(self,html_str):
        return re.findall(r'<h2><a target="_blank" href="(.*?)">(.*?)</a></h2>',html_str,re.S)

    def save(self,extract_content,page):
        path = "guokr第{}页.txt"
        for content in extract_content:
            with open(path.format(page),'a',encoding='utf-8') as f:
                f.write(json.dumps(content,ensure_ascii=False,indent=1))
    def run(self):
#1.url_list
        url_list =self.get_url_list()
#2.遍历，发送请求,获取响应parse_url
        for url in url_list:
            html_str = self.parse_url(url)
#3.提取
            extract_content = self.extract_content(html_str)
#4.save
            page = url_list.index(url)+1
            self.save(extract_content,page)

if __name__ == '__main__':
    guokr = Guokr()
    guokr.run()
