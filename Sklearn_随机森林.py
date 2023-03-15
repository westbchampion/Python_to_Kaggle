import pandas as pd


#使用pandas，读取fashion_mnist的训练和测试数据文件。
#读取文件使用相对路径地址
train_data = pd.read_csv('../datasets/fashion_mnist/fashion_mnist_train.csv')
test_data = pd.read_csv('../datasets/fashion_mnist/fashion_mnist_test.csv')
#从训练数据中，拆解出训练特征和类别标签。
X_train = train_data[train_data.columns[1:]]
y_train = train_data['label']

#从测试数据中，拆解出测试特征和类别标签。
X_test = test_data[train_data.columns[1:]]
y_test = test_data['label']
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#初始化随机森林分类器模型，并明确设定超参数。
rfc = RandomForestClassifier(n_estimators=10, random_state=2022)

#使用训练数据，训练随机森林分类器模型。
rfc.fit(X_train, y_train)

#使用训练好的分类模型，依据测试数据的特征，进行类别预测。
y_predict = rfc.predict(X_test)

#评估分类器的准确率。
print('Scikit-learn的随机森林分类器在fashion_mnist测试集上的准确率为：%.2f%%。' %(accuracy_score(y_test, y_predict) * 100))

