import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片标题中文显示
plt.rcParams['axes.unicode_minus'] = False

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, [0, 1, 2]]  # 选择萼片长度、萼片宽度和花瓣长度三个特征
y = iris.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# 生成模型
model = LogisticRegression(max_iter=1000, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 测试模型并计算准确率
test_accuracy = model.score(X_test, y_test) * 100
train_accuracy = model.score(X_train, y_train) * 100
print(f"逻辑回归测试集准确率: {test_accuracy:.6f}%")
print(f"逻辑回归训练集准确率: {train_accuracy:.6f}%")

# 准备绘制决策边界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # 横轴范围
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # 纵轴范围
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),  # 创建网格
                     np.arange(y_min, y_max, 0.01))

# 预测网格上的每一点
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # 平面上的点输入模型
Z = Z.reshape(xx.shape)  # 转换为网格形状

# 绘制决策边界
plt.figure(figsize=(10, 6))  # 设置画布大小
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)  # 决策边界的填充颜色

# 绘制训练集数据点
scatter_train = plt.scatter(X_train[:, 0], X_train[:, 1],  # 训练集特征
                            c=y_train, edgecolor="k", s=80, marker='o', cmap=plt.cm.Paired,
                            label="训练集数据")  # 圆形标记
# 绘制测试集数据点
scatter_test = plt.scatter(X_test[:, 0], X_test[:, 1],  # 测试集特征
                           c=y_test, edgecolor="k", s=120, marker='^', cmap=plt.cm.Paired,
                           label="测试集数据")  # 三角形标记

# 图例和标签
plt.xlabel("萼片长度(cm)")
plt.ylabel("花瓣宽度(cm)")
plt.title(f"逻辑回归, 训练集准确率：{train_accuracy:.6f}%，测试集准确率: {test_accuracy:.6f}%")
plt.legend(loc="upper left", fontsize=12)  # 添加图例
plt.colorbar(scatter_train)  # 添加颜色条，用于标识类别
plt.savefig('./Pictures/Logistic Regression.png')
