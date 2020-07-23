import numpy
import requests
tieba_name = input(":")
url_temp = "https://tieba.baidu.com/f?kw="+tieba_name+"&ie=utf-8&pn={}"
url_list = [url_temp.format(i*50) for i in range(2)]
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400'}
for url in url_list:
    resp = requests.get(url,headers=headers)
    file_path = "{}_第{}页.html".format(tieba_name,url_list.index(url)+1)
    with open(file_path, 'w',encoding='utf-8')as file:
        file.write(resp.content.decode())
        
