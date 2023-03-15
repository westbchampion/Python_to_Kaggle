#读取sklearn上的数据集
from sklearn.datasets import load_digits

X,y=load_digits(return_X_y=True)

from sklearn.model_selection import train_test_split

#拆分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=2022)

#调用决策树模型
from sklearn.tree import DecisionTreeClassifier

#初始化决策树模型
dtc=DecisionTreeClassifier()

#训练决策树模型
dtc.fit(X_train,y_train)

#进行数据预测
y_predict=dtc.predict(X_test)

from sklearn.metrics import accuracy_score


print ('Scikit-learn的决策树分类器在使用原始特征的digits测试集上的准确率为：%.2f%%。' %(accuracy_score(y_test, y_predict) * 100))
