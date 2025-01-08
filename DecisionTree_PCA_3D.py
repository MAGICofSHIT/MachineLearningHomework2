import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 加载葡萄酒数据集
wine = load_wine()
X = wine.data  # 特征
y = wine.target  # 标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维到3维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 拆分训练集和测试集
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=420)

# 初始化决策树分类器
base_learner_PCA = DecisionTreeClassifier(max_depth=3, random_state=42)

# 训练PCA后的决策树模型
base_learner_PCA.fit(X_train_pca, y_train_pca)

# 输出PCA后决策树模型在训练集和测试集上的准确率
train_accuracy_tree_pca = base_learner_PCA.score(X_train_pca, y_train_pca) * 100
test_accuracy_tree_pca = base_learner_PCA.score(X_test_pca, y_test_pca) * 100
print(f"PCA降维后决策树测试集准确率: {test_accuracy_tree_pca:.6f}%")
print(f"PCA降维后决策树训练集准确率: {train_accuracy_tree_pca:.6f}%")

# 准备绘制PCA后的三维决策边界
x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
z_min, z_max = X_pca[:, 2].min() - 0.5, X_pca[:, 2].max() + 0.5

xx, yy, zz = np.meshgrid(
    np.arange(x_min, x_max, 0.5),
    np.arange(y_min, y_max, 0.5),
    np.arange(z_min, z_max, 0.5)
)

# 预测网格上的每一点
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
Z = base_learner_PCA.predict(grid_points)
Z = Z.reshape(xx.shape)

# 绘制三维决策边界和数据点
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制训练集数据点
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train_pca,
           marker='o', s=80, cmap=plt.cm.Paired, label="训练集数据")

# 绘制测试集数据点
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_test_pca,
           marker='^', s=120, cmap=plt.cm.Paired, label="测试集数据")

# 绘制决策边界 (透明点表示分类区域)
ax.scatter(xx, yy, zz, c=Z, alpha=0.03, cmap=plt.cm.Paired)

# 设置标签
ax.set_xlabel("PCA特征 1")
ax.set_ylabel("PCA特征 2")
ax.set_zlabel("PCA特征 3")
ax.set_title(f"PCA后决策树分类, 训练集准确率: {train_accuracy_tree_pca:.6f}%，测试集准确率: {test_accuracy_tree_pca:.6f}%")
ax.legend(loc="best", fontsize=12)
plt.savefig('./Pictures/DecisionTreeClassifier_after_PCA_3D.png')
plt.show()
