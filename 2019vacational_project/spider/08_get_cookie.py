import requests
url ='http://www.baidu.com'

response = requests.get(url)
print(response.cookies)
print(type(response.cookies))
cookie = requests.utils.dict_from_cookiejar(response.cookies)
print(cookie)