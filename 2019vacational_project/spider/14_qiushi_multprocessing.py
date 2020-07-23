import requests
from lxml import etree
#from queue import Queue
#import threading
from multiprocessing import Process
from multiprocessing import JoinableQueue as Queue
import time
class Qiushi:
    def __init__(self):
        self.start_url = "https://www.qiushibaike.com/8hr/page/{}/"
        self.add_url = "https://www.qiushibaike.com"
        self.add1_url = 'https:'
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}
        self.url_queue = Queue()
        self.html_queue = Queue()
        self.content_list_queue = Queue()
    def get_url_list(self):
        for i in range(1,14):
            self.url_queue.put(self.start_url.format(i))

    def parse_url(self):
        while True:#不加while时，线程只进行一次，加了以后则无限循环，设置成守护线程时，会随主而结束
            url = self.url_queue.get()
            resp = requests.get(url, headers=self.headers)
            print(resp.status_code)
            self.html_queue.put(resp.content.decode())
            self.url_queue.task_done()#让队列计数-1

    def contract_content(self):
        while True:
            html_str = self.html_queue.get()
            content_list = []
            html = etree.HTML(html_str)
            li_list = html.xpath('//a[contains(@class,"recmd-left")]')
            print(len(li_list))

            for li in li_list:
                item = {}  # 每次要重置归0，否则导致20个数一样
                item['href'] = self.add_url + li.xpath('./@href')[0] if len(li.xpath('./@href')) > 0 else None
                item['img_src'] = self.add1_url + li.xpath('./img/@src')[0] if len(li.xpath('./img/@src')) > 0 else None
                item['title'] = li.xpath('./img/@alt')[0] if len(li.xpath('./img/@alt')) > 0 else None

                content_list.append(item)
            self.content_list_queue.put(content_list)
            self.html_queue.task_done()


    def save(self):
        while True:
             content_list = self.content_list_queue.get()

             for content in content_list:
                 pass
             self.content_list_queue.task_done()

    def run(self):
        thread_list=[]
        # 1.初url
        t_url=Process(target=self.get_url_list)
        thread_list.append(t_url)

        # 2.发送讲求，获取响应
        for i in range(3):
            t_parse=Process(target=self.parse_url)
            thread_list.append(t_parse)
            # 3.提取数据
        t_content=Process(target=self.contract_content)
        thread_list.append(t_content)

            # 4.save
        t_save=Process(target=self.save)
        thread_list.append(t_save)
    # 5.next——url
        for process in thread_list:#把子线程设置成守护线程
            process.daemon = True
            process.start()
        for q in [self.url_queue, self.html_queue, self.content_list_queue]:
            q.join()#让主线程阻塞，等队列计数为0
if __name__ == '__main__':
    t1=time.time()
    qiushi = Qiushi()
    qiushi.run()
    print(time.time()-t1)