import pandas as pd

#默认索引的Series（Excel的列）
s=pd.Series(data=[1,2,3])



#指定自定义索引的Series
s2=pd.Series(data=[1,2,3],index=['a',5,6])

"""
#自定义索引的Series
d3={
    'a':1,'b':2,'c':3
}

s3=pd.Series(d3)
"""

#打印出来
print(s)
print(s2)
#print(s3)



#默认索引的DataFrame(Excel的表)
#写表头和列表的感觉
d1={
    '国家': ['中国','美国','日本'],
    '人口':['13','8','1']
}

Data1=pd.DataFrame(d1)
print(Data1)

#自定义索引，在DataFrame里定义
Data2=pd.DataFrame(d1,index=['a','b','c'])
print(Data2)

#pandas读写文件
#文件的相对地址写法如下

'''
df=pd.read_csv('从零开始Kaggle竞赛/test_data.csv')
'''
#数据分析常用功能

#生成DataFrame表
#写表头，写列元素
d={
    '国家':['中国','美国','日本'],
    '人口':['13','8','1']
}

#生成表格
df=pd.DataFrame(d)

#指定位置填写
#添加一行列表元素
#loc位置+字典结构
df.loc[3]={
    '国家':'俄罗斯',
    '人口':'6'
}

print(df)

#利用append追加DataFrame
d2={
    '国家':['意大利','德国'],
    '人口':['2','3']
}
df2=pd.DataFrame(d2)

new_df=df.append(df2)
print(new_df)

#concat用于DataFrame拼接

#竖着写
#创建新列，并使用Series赋值
df['国土面积']=pd.Series([960,800,200,100])

print(df)

#删除数据,需要重新赋值
#横着删
print(df.drop([0,1]))


#竖着删
print(df.drop(columns='国土面积'))


#loc横着选
print(df.loc[3])

#名字竖着选
print(df["国家"])

#切片选择多行
print(df[0:3])
print(df)
#修改具体元素（行列）（不需要赋值）
df.loc[2,'人口']=1.29
print(df)
#修改整列
#df.loc[:,'国土面积']=[960.700,937,800]
#修改整行
df.loc[1]=['冰岛',1.26,100]
print(df)
