import re
import urllib.request

def getcontent(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0(Windows NT 10.0;Win64;x64;rv:66.0)Gecko/20100101 Firefox/66.0')
    data = urllib.request.urlopen(req).read().decode('utf-8')

    pattern1 = '<a href="/(.*?)" target="_blank" title=".*?">'
    urlList = re.compile(pattern1).findall(data)
    pattern2 = '<a href="/.+?" target="_blank">(.*?)</a>'
    pattern3 = '<a href=".*?" target="_blank" title="(.*?)">'
    sourceList = re.compile(pattern2).findall(data)
    titleList = re.compile(pattern3).findall(data)
    authorList = []
    totalUrlList = []
    timeList = []
    info = []
    for url in urlList:
        url = "https://bbs.hupu.com/topic-postdate" + url
        totalUrlList.append(url)
        html = urllib.request.urlopen(url).read().decode('utf-8')
        pattern = '<a class="u" target="_blank" href=".*?">(.*?)</a>'
        pattern4 = '<span class="stime">(.*?)</span>'
        aulist = re.compile(pattern).findall(html)
        tiList = re.compile(pattern4).findall(html)
        authorList.append(aulist[0])
        timeList.append(tiList[0])
    info.append(totalUrlList)
    info.append(sourceList)
    info.append(titleList)
    info.append(authorList)
    info.append(timeList)
    return info


if __name__ == '__main__':
    url = "https://bbs.hupu.com/topic-postdate"
    info = getcontent(url)
    totalurlList = info[0]
    sourceList = info[1]
    titleList = info[2]
    authorList = info[3]
    timeList = info[4]
    length = len(totalurlList)
    for i in range(length):
        str = "标题：" + titleList[i] + " " + "作者：" + authorList[i] + " " + "URL：" + totalurlList[i] + " " + "发布时间：" + \
              timeList[i] + " " + "帖子来源：" + sourceList[i]
        print(str)
