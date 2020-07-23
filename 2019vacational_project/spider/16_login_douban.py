import time
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("http://www.douban.com")
driver.find_element_by_class_name("account-tab-account").click()
driver.find_element_by_id('username').send_keys('1111111')
driver.find_element_by_id('password').send_keys('2222')
time.sleep(4)
driver.quit()