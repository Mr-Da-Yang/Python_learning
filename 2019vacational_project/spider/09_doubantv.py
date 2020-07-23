import requests
import json


class Douban_spider:
    def __init__(self):
        self.url = 'https://m.douban.com/rexxar/api/v2/subject_collection/tv_american/items?os=ios     &for_mobile=1&start={}&count=18&loc_id=108288&_=1563850510540'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/56.0.2924.75 Mobile/14E5239e Safari/602.1',

            'Referer': 'https://m.douban.com/tv/american'}

    def url_list(self):
        url_list = [self.url.format(i * 18) for i in range(3)]
        return url_list

    def parse_url(self, url):
        response = requests.get(url, headers=self.headers)
        json_str = response.content.decode()
        lists = json.loads(json_str)['subject_collection_items']
        return lists

    def save(self, url_list):
        for url in url_list:
            lists = self.parse_url(url)
            save_path = '美剧-第{}页.txt'
            with open(save_path.format(self.url_list().index(url) + 1), 'a', encoding='utf-8') as f:
                for list in lists:
                    f.write(json.dumps(list, ensure_ascii=False))
                    f.write("\n")

    def run(self):
        # 1.url_list
        url_list = self.url_list()
        # 2. parse_url

        # 3. save
        self.save(url_list)


if __name__ == '__main__':
    spider = Douban_spider()
    spider.run()
