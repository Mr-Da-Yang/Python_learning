import requests
from lxml import etree
class Qiushi:
    def __init__(self):
        self.start_url = "https://www.qiushibaike.com/8hr/page/1/"
        self.add_url ="https://www.qiushibaike.com"
        self.add1_url='https:'
        self.headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}


    def parse_url(self,url):
        resp = requests.get(url,headers=self.headers)
        return resp.content.decode()

    def contract_content(self,html_str):

        content_list =[]
        html=etree.HTML(html_str)
        li_list = html.xpath('//a[contains(@class,"recmd-left")]')
        print(len(li_list))

        for li in li_list:
            item={}#每次要重置归0，否则导致20个数一样
            item['href'] = self.add_url+li.xpath('./@href')[0] if len(li.xpath('./@href')) >0 else None
            item['img_src'] =self.add1_url+ li.xpath('./img/@src')[0] if len(li.xpath('./img/@src'))>0 else None
            item['title'] = li.xpath('./img/@alt')[0] if len( li.xpath('./img/@alt'))>0 else None

            content_list.append(item)

        html_next_url = html.xpath("//span[@class='next']")[0]
        next_url = self.add_url+ html_next_url.xpath('../@href')[0] if len(html_next_url.xpath('../@href'))> 0  else None
        print(next_url)
        return content_list , next_url

    def save(self,content_list):
        pass








    def run(self):
    # 1.初url
        next_url = self.start_url
        while next_url is not None:

    #2.发送讲求，获取响应
            html_str = self.parse_url(next_url)
    #3.提取数据
            content_list, next_url = self.contract_content(html_str)
    #4.save
            self.save(content_list)
    #5.next——url

if __name__ == '__main__':
    qiushi = Qiushi()
    qiushi.run()
