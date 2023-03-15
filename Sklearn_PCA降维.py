#导入iris数据集
from sklearn.datasets import load_iris

#读取iris数据集
X,y=load_iris(return_X_y=True)

from sklearn.preprocessing import StandardScaler

#初始化标准处理器
ss=StandardScaler()

#标准化数据特征
X=ss.fit_transform(X)
#初始化主成分分析器进行降维设定
from sklearn.decomposition import PCA
#设定为降到2维
pca=PCA(n_components=2)
#对数据进行降维处理
X=pca.fit_transform(X)

#画图可视化库初始化
from matplotlib import pyplot as plt

colors=['red','blue','green']
markers=['o','^','s']

plt.figure(dpi=150)

#遍历降维数据并可视化
for i in range(len(X)):
    plt.scatter(X[i,0],X[i,1],c=colors[y[i]],marker=markers[y[i]])

plt.show()
