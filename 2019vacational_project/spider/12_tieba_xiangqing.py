import requests
from lxml import etree


class TiebaSpider:
    def __init__(self, tieba_name):
        self.start_url = 'https://tieba.baidu.com/f?ie=utf-8&kw={}&fr=search&pn=0&'.format(tieba_name)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}
        self.url = 'https://tieba.baidu.com'

    def parse_url(self, url):
        reponse = requests.get(url, headers=self.headers)
        print(reponse.status_code)
        return reponse.content.decode()

    def get_content_list(self, html_str):
        html = etree.HTML(html_str)
        # 选/body/div/div这样更加准确，用contain,因为class=i/class=it 都是我们想要的,这里必需是""
        div_list = html.xpath('//body/div/div[contains(@class,"i")]')
        content_list = []
        for div in div_list:
            item = {}
            item['href'] = self.url + div.xpath('./a/@href')[0] if len(div.xpath("./a/@href")) > 0 else None
            item['title'] = div.xpath('./a/text()')[0]
            item['img_list'] = self.get_img_list(item['href'],[])
            content_list.append(item)

        # 提取next_url
        next_url = self.url + html.xpath('//a[text()="下一页"]/@href')[0] if html.xpath('//a[text()="下一页"]/@href')[
                                                                               0] > 0 else None
        return content_list, next_url  ####返回两个数据
    def get_img_list(self,detail_ur,img_list):
        #1.发送请求，获取响应
        detail_html_str=self.parse_url(detail_ur)

    #2.提取数据
        detail_html = etree.HTML(detail_html_str)
        img_list += detail_html.xpath('//img[@class="DBE_Image"]/@src')
    #3.详情页next_url
        next_url = detail_html.xpath('//a[text()="下一页"]/@href')[0] if detail_html.xpath('//a[text()="下一页"]/@href')[
                                                                              0] > 0 else None
        if next_url is not None:#当存在详情页下一页时，请求
           return self.get_img_list(next_url,img_list)
        else:
            img_list =[requests.utils.unquote(i).split('scree')[-1] for i in img_list]
            return img_list

    def save_content_list(self, content_list):
        for content in content_list:
            print(content)

    def run(self):
        next_url = self.start_url
        while next_url is not None:
            # 1.先获得第1个url--获取其信息后，while再第2，3，
            # 因为总的页数是不确定的，所以不好先生成url——list
            # start_url

            # 2.发送请求，获取响应
            html_str = self.parse_url(next_url)
            # 3.提取数据
            content_list, next_url = self.get_content_list(html_str)

            # 4.保存
            self.save_content_list(content_list)
            # 5.获取next_url,循环2-5

if __name__ == '__main__':
    tieba = TiebaSpider('李毅')
    tieba.run()