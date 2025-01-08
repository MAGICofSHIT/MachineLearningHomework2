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

# PCA降维
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
print(f"PCA降维后决策树训练集准确率: {train_accuracy_tree_pca:.6f}%")
print(f"PCA降维后决策树测试集准确率: {test_accuracy_tree_pca:.6f}%")

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=420)

# 构造协方差矩阵，得到特征向量和特征值
cov_matrix = np.cov(X_train.T)
eigen_val, eigen_vec = np.linalg.eig(cov_matrix)

# 解释方差比
tot = sum(eigen_val)  # 总特征值和
var_exp = [(i / tot) for i in sorted(eigen_val, reverse=True)]  # 计算解释方差比，降序
cum_var_exp = np.cumsum(var_exp)  # 累加方差比率
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='独立解释方差')
plt.step(range(1, 14), cum_var_exp, where='mid', label='累加解释方差')
plt.ylabel("解释方差率")
plt.xlabel("主成分索引")
plt.legend(loc='right')
plt.savefig('./Pictures/PCA_Variance_Ratio.png')
plt.show()

# 准备绘制PCA后决策树的决策边界
x_min_pca, x_max_pca = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min_pca, y_max_pca = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx_pca, yy_pca = np.meshgrid(np.arange(x_min_pca, x_max_pca, 0.01),
                              np.arange(y_min_pca, y_max_pca, 0.01))

# 预测网格上的每一点
Z_tree_pca = base_learner_PCA.predict(np.c_[xx_pca.ravel(), yy_pca.ravel()])
Z_tree_pca = Z_tree_pca.reshape(xx_pca.shape)

# 绘制PCA后决策树决策边界图像
plt.figure(figsize=(9, 6))
plt.contourf(xx_pca, yy_pca, Z_tree_pca, alpha=0.8, cmap=plt.cm.Paired)

# 绘制训练集数据点
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, edgecolor="k", s=80,
            marker='o', cmap=plt.cm.Paired, label="训练集数据")

# 绘制测试集数据点
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, edgecolor="k", s=120,
            marker='^', cmap=plt.cm.Paired, label="测试集数据")

# 图例和标签
plt.xlabel("PCA特征 1")
plt.ylabel("PCA特征 2")
plt.title(f"PCA后决策树分类, 训练集准确率: {train_accuracy_tree_pca:.6f}%，测试集准确率: {test_accuracy_tree_pca:.6f}%")
plt.legend(loc="upper left", fontsize=12)
plt.colorbar()  # 添加颜色条
plt.savefig('./Pictures/DecisionTreeClassifier_after_PCA.png')
plt.show()