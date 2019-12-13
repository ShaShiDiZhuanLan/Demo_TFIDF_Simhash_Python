# encoding: utf-8
"""
Author: 沙振宇
CreateTime: 2018-12-25
UpdateTime: 2019-12-12
Info: 正则表达式
"""
import re

# 正则匹配处理
def run(content):
    content = content.replace("\n", "")
    content = content.replace("&nbsp;", "")
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
    for url in urls:
        content = content.replace(url, "")
    html_all = re.findall('<.*?>', content)
    for html in html_all:
        content = content.replace(html, "")

    ss = re.findall('[\n*\r\u4e00-\u9fa5|a-zA-Z0-9]', content)
    content = "".join(ss)
    return content

if __name__=="__main__":
    testdict = {"1.app是啥": "", "2.tests. The test code": "", "3.itle>无标题": "", "4.何东西($)。下面": "", "5.[[:": "",
                "6.^\\\/\^": "", "7.^[a-z][0-9]$": "", "8.[0-9] //匹配": "", "9.行，\n表示回车。其他": "", "10.": ""}
    for index in testdict.keys():
        data = run(index)
        print(index,"----------------------",data)