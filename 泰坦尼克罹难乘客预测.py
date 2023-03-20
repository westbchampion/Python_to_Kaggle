import pandas as pd

#读取训练集和测试集
train_data=pd.read_csv('从零开始Kaggle竞赛//titanic//train.csv')

test_data=pd.read_csv('从零开始Kaggle竞赛//titanic//test.csv')

#查询训练集表头
train_data.info()
#891个乘客与生存数据
test_data.info()
#418个乘客与生存数据

#数据预处理
def data_prepross(df):
    
    #丢弃缺失值较多的Cabin特性与共性较弱的文本特征,axis=1
    df=df.drop(['Cabin','Ticket','Name'],axis=1)

    #填充Age、Fare、Embarked特征
    #数值型特征使用平均数或中位数进行填充
    #类别型特征使用众数对缺失值进行填充
    df=df.fillna({
        'Age':df['Age'].median(),
        'Fare':df['Fare'].mean(),
        'Embarked':df['Embarked'].value_counts().idxmax()
    })
    return df

train_data=data_prepross(train_data)
test_data=data_prepross(test_data)


train_data.info()
test_data.info()

#选择训练集需训练数据为去除掉Survived和PassengeiID的数据
X_train=train_data.drop(['Survived','PassengerId'],axis=1)
print(X_train)
#选择训练集目标数据（预测数据）为Survived数据，用于训练模型
y_train=train_data['Survived']
print(y_train)
#测试集数据为去除掉PassengerId的测试及数据
X_test=test_data.drop(['PassengerId'],axis=1)

#X_train和X_test数据对齐

#获得训练和测试集中的数值型特征
num_X_train=X_train[['Age','Fare','SibSp','Parch']].values
num_X_test=X_test[['Age','Fare','SibSp','Parch']].values

#获得训练和测试集中的类别型特征，并转换为独热编码

from sklearn.preprocessing import OneHotEncoder

#初始化独热编码器
ohe=OneHotEncoder()
#获得训练和测试集中的类别型特征，并转换为独热编码
cate_X_train=ohe.fit_transform(X_train[['Pclass','Sex','Embarked']]).todense()
cate_X_test=ohe.fit_transform(X_test[['Pclass','Sex','Embarked']]).todense()

import numpy as np

#将数值特征与类别特征的独热编码进行拼接
X_train=np.concatenate([num_X_train,cate_X_train],axis=1)
X_test=np.concatenate([num_X_test,cate_X_test],axis=1)

#使用随机森林分类器，进行交叉验证与超参数寻优
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

parameters={
    'n_estimators':[10,50,100],
    'criterion':['gini','entrophy']
}

rfc = RandomForestClassifier()

#使用GridSearchCV函数进行交叉验证，得到最优超参数和最佳准确率
clf=GridSearchCV(rfc,parameters,scoring='accuracy',n_jobs=4)


#X_train为需要的训练数据
#y_train为训练出来的目标结果
#不能写反
clf.fit(X_train,y_train)

print('最优超参数为:%s'%clf.best_params_)
print('交叉验证得出的最佳准确率为：%f'%clf.best_score_)

#使用最佳模型进行类别预测
#X_test为测试所用数据，y_predict为预测所得的目标结果
y_predict=clf.predict(X_test)


#X_train和X_test同时进行数据预处理
#X_train与y_train同时进行fit函数训练模型
#X_test用于预测结果为y_predict

#DataFrame化提交结果后保存并提交+


submission=pd.DataFrame(
    {
        'PassengerId':test_data['PassengerId'],
        'Survived':y_predict
    }
)

submission.to_csv('从零开始Kaggle竞赛//submission.csv',index=False)


