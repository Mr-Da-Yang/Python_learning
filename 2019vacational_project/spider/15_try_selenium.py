from selenium import webdriver
import time
driver = webdriver.Chrome()
#driver.set_window_size(1800,500)
#driver.maximize_window()
driver.get("http://www.baidu.com")
driver.find_element_by_class_name("s_ipt").send_keys('python')#输入内容
driver.find_element_by_id("su").click()#点击元素
#driver.save_screenshot('baidu.png')
#print(driver.page_source)#获取element也就是网页源码，response获取的是url响应
print(driver.current_url)
print(driver.get_cookies())
time.sleep(2)
#driver.close()
driver.quit()