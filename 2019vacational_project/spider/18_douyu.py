import requests
from lxml import etree
class Douyu:
    def __init__(self):
        self.start_url = "https://www.douyu.com/directory/all"
        self.headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"
}
        self.add_url = "https://www.douyu.com"
    def parse_url(self,url):
        res = requests.get(url, headers=self.headers)
        return res.content.decode()

    def contract_content(self,content):
        html = etree.HTML(content)
        li_list = html.xpath("//a[@class='DyListCover-wrap']")
        content_list=[]
        for li in li_list:
            item = {}
            item["name"]=li.xpath("./div[2]/div[2]/h2/text()")[0]
            item["index"]=li.xpath("./div[2]/div[2]/span/text()")[0]
            item["url"]= self.add_url + li.xpath("./@href")[0]
            content_list.append(item)
        return content_list
    def save(self,content_list):
        print(content_list)




    def run(self):
#1.start_utl
        url = self.start_url
#2.发送数据获取响应
        content=self.parse_url(url)
#3.提取数据
        content_list = self.contract_content(content)

#4.保存
        self.save(content_list)

if __name__ == '__main__':
    douyu = Douyu()
    douyu.run()