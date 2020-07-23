import requests
url ='http://www.renren.com/941954027/profile'
headers ={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400",

           }
Cookie='anonymid=jy89kr1l5x0mxw; depovince=GW; jebecookies=880db678-fdac-4b91-80fb-68b2f73d24b1|||||; _r01_=1; ick_login=5f94598e-58a3-4c77-93c1-8842ba84f096; _de=7A7A02E9254501DA6278B9C75EAEEB7A; p=7b9fa292189016a5f2e4ce302c2f04397; first_login_flag=1; ln_uact=13146128763; ln_hurl=http://hdn.xnimg.cn/photos/hdn421/20181202/2020/main_AJKu_0a9a00001c18195a.jpg; t=330effc2d3f09dc1fa04dceef4d60c807; societyguester=330effc2d3f09dc1fa04dceef4d60c807; id=941954027; xnsid=fc5667bd; ver=7.0; loginfrom=null; jebe_key=04ee090a-3bb0-4916-92a3-ab80332419ef%7C6f8d20f6f9af5aad656d98d31d7800f4%7C1563429544106%7C1%7C1563429547989; wp_fold=0; wp=0'
cookie_dic = {i.split('=')[0]:i.split('=')[1] for i in Cookie.split('; ')}
response = requests.get(url, headers=headers,cookies=cookie_dic)

with open ('02zhoudawei.html','w',encoding='utf-8')as f:
    f.write(response.content.decode())