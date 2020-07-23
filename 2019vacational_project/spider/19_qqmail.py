from selenium import webdriver
import time
driver = webdriver.Chrome()
driver.get("https://mail.qq.com/cgi-bin/loginpage")
driver.switch_to.frame('login_frame')
driver.find_element_by_id("u").send_keys("2285322137")
time.sleep(3)
driver.quit()