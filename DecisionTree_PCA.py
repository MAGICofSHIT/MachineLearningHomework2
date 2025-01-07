import matplotlib.pyplot as plt
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

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=420)

# 初始化决策树分类器（最大深度为 1）
base_learner = DecisionTreeClassifier(max_depth=2,random_state=42)

# 训练基础决策树模型
base_learner.fit(X_train, y_train)

# 输出基础决策树模型在训练集和测试集上的准确率
train_accuracy_tree = base_learner.score(X_train, y_train) * 100
test_accuracy_tree = base_learner.score(X_test, y_test) * 100
print(f"决策树测试集准确率: {test_accuracy_tree:.6f}%")
print(f"决策树训练集准确率: {train_accuracy_tree:.6f}%")

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 拆分训练集和测试集
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=420)

# 训练PCA后的决策树模型
base_learner.fit(X_train_pca, y_train_pca)

# 输出PCA后决策树模型在训练集和测试集上的准确率
train_accuracy_tree_pca = base_learner.score(X_train_pca, y_train_pca) * 100
test_accuracy_tree_pca = base_learner.score(X_test_pca, y_test_pca) * 100
print(f"PCA降维后决策树测试集准确率: {test_accuracy_tree_pca:.6f}%")
print(f"PCA降维后决策树训练集准确率: {train_accuracy_tree_pca:.6f}%")

# # 准备绘制决策树的决策边界
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                      np.arange(y_min, y_max, 0.01))
#
# # 预测网格上的每一点
# Z_tree = base_learner.predict(np.c_[xx.ravel(), yy.ravel()])
# Z_tree = Z_tree.reshape(xx.shape)
#
# # 准备绘制PCA后决策树的决策边界
# x_min_pca, x_max_pca = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
# y_min_pca, y_max_pca = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
# xx_pca, yy_pca = np.meshgrid(np.arange(x_min_pca, x_max_pca, 0.01),
#                               np.arange(y_min_pca, y_max_pca, 0.01))
#
# # 预测网格上的每一点
# Z_tree_pca = base_learner.predict(np.c_[xx_pca.ravel(), yy_pca.ravel()])
# Z_tree_pca = Z_tree_pca.reshape(xx_pca.shape)
#
# # 图像绘制和保存
# # 绘制基础决策树决策边界图像
# plt.figure(figsize=(12, 7))
# plt.contourf(xx, yy, Z_tree, alpha=0.8, cmap=plt.cm.Paired)
#
# # 绘制训练集数据点
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", s=80,
#             marker='o', cmap=plt.cm.Paired, label="训练集数据")
#
# # 绘制测试集数据点
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k", s=120,
#             marker='^', cmap=plt.cm.Paired, label="测试集数据")
#
# # 图例和标签
# plt.xlabel("特征 1")
# plt.ylabel("特征 2")
# plt.title(f"基础决策树分类, 训练集准确率: {train_accuracy_tree:.6f}%，测试集准确率: {test_accuracy_tree:.6f}%")
# plt.legend(loc="upper left", fontsize=12)
# plt.colorbar()  # 添加颜色条
# plt.savefig('./Pictures/DecisionTreeClassifier_before_PCA.png')
# plt.show()
#
# # 绘制PCA后决策树决策边界图像
# plt.figure(figsize=(12, 7))
# plt.contourf(xx_pca, yy_pca, Z_tree_pca, alpha=0.8, cmap=plt.cm.Paired)
#
# # 绘制训练集数据点
# plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, edgecolor="k", s=80,
#             marker='o', cmap=plt.cm.Paired, label="训练集数据")
#
# # 绘制测试集数据点
# plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, edgecolor="k", s=120,
#             marker='^', cmap=plt.cm.Paired, label="测试集数据")
#
# # 图例和标签
# plt.xlabel("PCA特征 1")
# plt.ylabel("PCA特征 2")
# plt.title(f"PCA后决策树分类, 训练集准确率: {train_accuracy_tree_pca:.6f}%，测试集准确率: {test_accuracy_tree_pca:.6f}%")
# plt.legend(loc="upper left", fontsize=12)
# plt.colorbar()  # 添加颜色条
# plt.savefig('./Pictures/DecisionTreeClassifier_after_PCA.png')
# plt.show()