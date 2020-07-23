import requests
url = 'http://www.sina.com.cn'
proxies = {'http':'http://120.237.14.198:53281'}
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}
response = requests.get(url, proxies=proxies,headers=headers)
print(response.status_code)