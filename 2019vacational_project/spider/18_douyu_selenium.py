#下一页没有url，所以用click，
from selenium import webdriver
import time
class Douyu:
    def __init__(self):
        self.start_url = "https://www.douyu.com/directory/all"

        self.driver = webdriver.Chrome()
        self.add_url = "https://www.douyu.com"

    def get_content_list(self):

        li_list = self.driver.find_elements_by_xpath("//a[@class='DyListCover-wrap']")

        content_list=[]
        for li in li_list:
            item={}
            # item["anchor"]=li.find_element_by_xpath("./div[2]/div[2]/h2").text
            # item["watch-num"]=li.find_element_by_xpath("./div[2]/div[2]/span").text
            item['url']=li.find_element_by_xpath(".").get_attribute("href")
            content_list.append(item)
        next_url = self.driver.find_elements_by_xpath("//a[@class='shark-pager-next']")
        next_url = next_url[0] if len(next_url)>0 else None
        return content_list , next_url

    def save(self,content_list):
        print(content_list)

    def run(self):

#1,url
        
#2,发送请求，获取响应
        self.driver.get(self.start_url)
        
    #3.提取数据
        content_list ,next_url = self.get_content_list()
    #4.save
        self.save(content_list)
    #5.next_url
        while next_url is not None:
            next_url.click()#因为第一次是.get方法，第二次是.click不一样，所以while后放/
            time.sleep(3)#再者页面没有完全加载.click没有等待就next，导致出现错误
            content_list, next_url = self.get_content_list()
            self.save(content_list)
if __name__ == '__main__':
    douyu = Douyu()
    douyu.run()