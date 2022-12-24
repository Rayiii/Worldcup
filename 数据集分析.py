import pandas as pd

# 导入数据
xls=pd.ExcelFile(r'D:\Rayi\大学\大学学习\大三课程\数据挖掘\世界杯\WorldCupMatches.xlsx')
Matches=xls.parse('WorldCupMatches')
Matches.head()
Matches.drop_duplicates(subset=['MatchID'], keep='first', inplace=True)
Matches.info()
# 由于观众人数缺失了一个数据，使用fillna函数对其填入观众的平均值
Matches['Attendance'] = Matches['Attendance'].fillna(int(Matches['Attendance'].mean()))

# 使用while循环遍历数据统计出主客队得分来判断胜负情况
i = 0
w = 0
p = 0
l = 0
while i < 836:
    if Matches['Home Team Goals'][i] > Matches['Away Team Goals'][i]:
        w += 1
    elif Matches['Home Team Goals'][i] == Matches['Away Team Goals'][i]:
        p += 1
    else:
        l += 1
    i += 1
print(w)
print(p)
print(l)
data = [w, p, l]

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure('Subplot',dpi=150)
plt.subplot(221)
x=range(len(data))
y=data
plt.title('主客场胜场数',fontsize=8)
plt.xlabel('对局',fontsize=8)
plt.ylabel('胜场数',fontsize=8)
plt.ylim((0,700))
plt.grid(axis='y', linestyle=':')
name_list=['主场胜','平局','客场胜']
plt.bar(x, y, 0.5,fc='b',alpha=0.5,tick_label=name_list,label='Sapltle 1')
for _x, _y in zip(x, y):
    plt.text(_x, _y, _y,ha='center', va='bottom', size=8)
plt.legend()

plt.subplot(222)
lables = ['主队','客队']
plt.title('主客场胜场得分箱线图',fontsize=8)
plt.xlabel('队伍',fontsize=8)
plt.ylabel('得分',fontsize=8)
plt.boxplot([Matches['Home Team Goals'],Matches['Away Team Goals']],sym='o',whis=1.5,labels=lables)
plt.grid(axis='y', linestyle=':')
plt.tight_layout()
plt.show()

groupbyyearDf=Matches.groupby('Year').sum()
groupbyyearDf=groupbyyearDf.reset_index()

x=groupbyyearDf['Year']
y=groupbyyearDf['Attendance']/10000
plt.figure('bar',dpi=120)
plt.title('观众总数')
plt.xlabel('年份')
plt.ylabel('观众人数(万)')
plt.grid(axis='y',linestyle=':')
plt.xticks(groupbyyearDf['Year'],rotation=45)
plt.plot(x,y,color='b')
plt.show()

GoalsDf=pd.DataFrame()
GoalsDf['GoalsHalf']=groupbyyearDf['Half-time Home Goals']+groupbyyearDf['Half-time Away Goals']
GoalsDf['GoalsHalf2']=groupbyyearDf['Home Team Goals']+groupbyyearDf['Away Team Goals']-GoalsDf['GoalsHalf']

x=groupbyyearDf['Year']
y1=GoalsDf['GoalsHalf2']
y2=GoalsDf['GoalsHalf']
plt.figure('bar',dpi=120)
plt.xticks(groupbyyearDf['Year'],rotation=45)
plt.grid(axis='y',linestyle=':')
plt.bar(x,y2,2,label='上半场')
plt.bar(x,y1,2,bottom=y2,label='下半场')
plt.xlabel('年份',fontsize=15)
plt.ylabel('总得分',fontsize=15)
plt.title('历届得分统计',fontsize=15)
plt.legend()
plt.show()

randDf=pd.DataFrame()
plt.figure('Subplot', facecolor='lightgray',figsize=(6, 5),dpi=150)
plt.subplot(221)
x=randDf['name']
y=randDf['Score']
plt.bar(x, y,fc='g',alpha=0.5)
plt.title('国家得分排名',fontsize=6)
plt.grid(axis='y',linestyle=':')
for _x,_y in zip(x,y):
    plt.text(_x,_y,_y,ha='center',va='bottom',size=8)
plt.ylim(0,300)
plt.ylabel('得分',fontsize=5)
plt.xticks(rotation=45,fontsize=5)
plt.yticks(fontsize=5)

AppearanceDf=pd.DataFrame()
plt.subplot(222)
x=AppearanceDf['index']
y=AppearanceDf['count']
plt.grid(axis='y',linestyle=':')
plt.bar(x, y,fc='g',alpha=0.5)
plt.ylim(0,150)
plt.title('国家出场频率',fontsize=6)
plt.ylabel('得分',fontsize=5)
for _x,_y in zip(x,y):
    plt.text(_x,_y,_y,ha='center',va='bottom',size=8)
plt.xticks(rotation=45,fontsize=5)
plt.yticks(fontsize=5)

AttenCountryDf=pd.DataFrame()
plt.subplot(223)
x=AttenCountryDf['Conutry']
y=AttenCountryDf['Atten']/10000
plt.title('每个国家队观众数量(万)',fontsize=8)
plt.ylabel('观众数量',fontsize=5)
plt.grid(axis='y',linestyle=':')
plt.ylim(0,700)
for _x, _y in zip(x, y):
    plt.text(_x, _y, '%.2f' % _y,ha='center',va='bottom', size=6)
plt.bar(x,y,color='g',alpha=0.5)
plt.xticks(rotation=45,fontsize=5)
plt.yticks(fontsize=5)

wincountDf=pd.DataFrame()
plt.subplot(224)
x=wincountDf['winner']
y=wincountDf['wincount']
plt.title('世界杯冠军次数',fontsize=6)
plt.ylabel('次数',fontsize=5)
plt.grid(axis='y',linestyle=':')
plt.bar(x,y,color='g')
plt.xticks(rotation=45,fontsize=5)
plt.yticks(fontsize=5)

plt.tight_layout()
plt.show