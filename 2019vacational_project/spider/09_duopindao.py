import requests
import json
class Douban:
    def __init__(self):
        self.url_temp = 'https://m.douban.com/rexxar/api/v2/subject_collection/tv_american/items?os=ios&for_mobile=1&start={}&count=18&loc_id=108288&_=1563850510540'
        self.url_temp_list =[
            {'url_temp':'https://m.douban.com/rexxar/api/v2/subject_collection/tv_american/items?os=ios&for_mobile=1&start={}&count=18&loc_id=108288&_=1563850510540',
             'Referer': 'https://m.douban.com/tv/american'},
            {'url_temp':'https://m.douban.com/rexxar/api/v2/subject_collection/tv_domestic/items?os=ios&for_mobile=1&&start={}&count=18&loc_id=108288&_=1563859628622',
             'Referer': 'https://m.douban.com/tv/chinese'}
        ]
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/56.0.2924.75 Mobile/14E5239e Safari/602.1',
            }



    def parse_url(self,url):
        print(url)
        resp = requests.get(url, headers=self.headers)
        json_str = resp.content.decode()

        return json_str

    def get_content_list(self,json_str):
        temp_dict = json.loads(json_str)
        return temp_dict['subject_collection_items']

    def save_content_list(self, content_list):
        with open("douban11.txt", 'a',encoding='utf-8')as f:
            for content in content_list:
                f.write(json.dumps(content,ensure_ascii=False))
                f.write('\n')

    def run(self):
        for url_temp in self.url_temp_list:
            self.headers.update({'Referer':url_temp['Referer']})
            num = 0
            while True:
        #1.start_url
                next_url=url_temp['url_temp'].format(num)
        #2.response
                json_str = self.parse_url(next_url)
        #3.提取
                content_list = self.get_content_list(json_str)
        #4.save
                self.save_content_list(content_list)
        #5.构造下一页url，循环2-5
                num +=18
                if len(content_list)<18:
                    break

if __name__ == '__main__':
    douban =Douban()
    douban.run()