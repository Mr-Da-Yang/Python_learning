from selenium import webdriver
import time
driver = webdriver.Chrome()
driver.get("https://movie.douban.com/top250")
# ret1 = driver.find_element_by_xpath('//div[@class="item"]')
# print(ret1)
# print("*"*100)
# ret2 = driver.find_elements_by_xpath('//div[@class="item"]')
# print(ret2)
# ret3 = driver.find_elements_by_xpath("//span[@class='title']")
# ret3=[i.text for i in ret3]
# print(ret3)
# ret4 = driver.find_elements_by_xpath("//span[@class='title']/..")
# ret4 =[i.get_attribute("href")for i in ret4]
# print(ret4)
# ret5 = driver.find_element_by_link_text("后页>").get_attribute("href")
# print(ret5)
ret6 = driver.find_element_by_partial_link_text("后页").get_attribute('href')
print(ret6)
time.sleep(4)
driver.quit()