import requests
url = "https://www.vmall.com/?cid=91895"
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}
response = requests.get(url, headers=headers)
print(response.content.decode())
#with open("baidu.png", "wb") as file:
    #file.write(response.content)
print(response.request.url)
print(response.url)
print(response.request.headers)
print(response.headers)