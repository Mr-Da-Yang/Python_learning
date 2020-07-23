import requests
session = requests.Session()
headers ={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}
post_url = "http://renren.com/PLogin.do"
post_data = {"email":'13146128763', 'password':'zhoudawei123'}
session.post(post_url,data=post_data,headers=headers)
profile_url = "http://www.renren.com/941954027/profile"
response = session.get(profile_url,headers=headers)
with open("renren.html", 'w',encoding='utf-8') as f:
    f.write(response.content.decode())