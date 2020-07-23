import requests
query_string = input(":")
params = {"wd": query_string}
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}
url = "http://www.baidu.com/s"
response = requests.get(url, params=params, headers=headers)
print(response.status_code)
print(response.request.url)