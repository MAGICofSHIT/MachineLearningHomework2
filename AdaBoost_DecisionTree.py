import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, [1, 3]]  # 选择花瓣长度和宽度两个特征
y = iris.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# 初始化 AdaBoost 模型
# 使用决策树作为弱学习器，设置最大深度为 1
base_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
adaboost_model = AdaBoostClassifier(estimator=base_learner, n_estimators=300, algorithm="SAMME", random_state=42)

# 训练模型
adaboost_model.fit(X_train, y_train)

# 测试模型并计算准确率
train_accuracy_ada = adaboost_model.score(X_train, y_train) * 100
test_accuracy_ada = adaboost_model.score(X_test, y_test) * 100
print(f"AdaBoost 测试集准确率: {test_accuracy_ada:.6f}%")
print(f"AdaBoost 训练集准确率: {train_accuracy_ada:.6f}%")

# 准备绘制决策边界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 预测网格上的每一点
Z_ada = adaboost_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_ada = Z_ada.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(12, 7))
plt.contourf(xx, yy, Z_ada, alpha=0.8, cmap=plt.cm.Paired)

# 绘制训练集数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", s=80,
            marker='o', cmap=plt.cm.Paired, label="训练集数据")

# 绘制测试集数据点
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k", s=120,
            marker='^', cmap=plt.cm.Paired, label="测试集数据")

# 图例和标签
plt.xlabel("花瓣长度(cm)")
plt.ylabel("花瓣宽度(cm)")
plt.title(f"AdaBoost 分类, 训练集准确率: {train_accuracy_ada:.6f}%，测试集准确率: {test_accuracy_ada:.6f}%")
plt.legend(loc="upper left", fontsize=12)
plt.colorbar()  # 添加颜色条
plt.savefig('./Pictures/AdaBoost_Classification.png')
plt.show()
