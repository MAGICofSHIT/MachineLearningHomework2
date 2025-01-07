import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine

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

# 用于存储决策树的训练和测试准确率
train_accuracies_decisionTree = []
test_accuracies_decisionTree = []

# 定义深度范围
depth_range = range(1, 11)

for depth in depth_range:
    # 初始化决策树分类器
    base_learner = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # 训练决策树模型
    base_learner.fit(X_train, y_train)

    # 输出基础决策树模型在训练集和测试集上的准确率
    train_accuracy_tree = base_learner.score(X_train, y_train) * 100
    test_accuracy_tree = base_learner.score(X_test, y_test) * 100
    print(f"决策树深度：{depth}")
    print(f"决策树测试集准确率: {test_accuracy_tree:.6f}%")
    print(f"决策树训练集准确率: {train_accuracy_tree:.6f}%\n")

    # 保存训练结果
    train_accuracies_decisionTree.append(train_accuracy_tree)
    test_accuracies_decisionTree.append(test_accuracy_tree)

# 绘制决策树训练集和测试集准确率曲线
plt.figure(figsize=(9, 6))
plt.plot(depth_range, train_accuracies_decisionTree, label="训练集准确率", marker='o')
plt.plot(depth_range, test_accuracies_decisionTree, label="测试集准确率", marker='o')
plt.xlabel("决策树深度")
plt.ylabel("准确率 (%)")
plt.title("不同深度决策树的训练集与测试集准确率对比")
plt.legend()
plt.grid(True)
plt.savefig('./Pictures/Accuracies_DecisionTreeClassifier.png')
plt.show()

# 弱学习器范围
number_estimators = range(1, 81)

# 用于adaboost的训练和测试准确率
train_accuracies_adaBoost = []
test_accuracies_adaBoost = []

for number in number_estimators:
    # 初始化 AdaBoost 模型
    base_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
    adaboost_model = AdaBoostClassifier(estimator=base_learner, n_estimators=number, algorithm="SAMME", random_state=42)

    # 训练 AdaBoost 模型
    adaboost_model.fit(X_train, y_train)

    # 测试 AdaBoost 模型并计算准确率
    train_accuracy_adaBoost = adaboost_model.score(X_train, y_train) * 100
    test_accuracy_adaBoost = adaboost_model.score(X_test, y_test) * 100
    print(f"弱学习器数量：{number}")
    print(f"AdaBoost测试集准确率: {test_accuracy_adaBoost:.6f}%")
    print(f"AdaBoost训练集准确率: {train_accuracy_adaBoost:.6f}%\n")

    # 保存训练结果
    train_accuracies_adaBoost.append(train_accuracy_adaBoost)
    test_accuracies_adaBoost.append(test_accuracy_adaBoost)

# 绘制adaBoost训练集和测试集准确率曲线
plt.figure(figsize=(9, 6))
plt.plot(number_estimators, train_accuracies_adaBoost, label="训练集准确率", marker='o')
plt.plot(number_estimators, test_accuracies_adaBoost, label="测试集准确率", marker='o')
plt.xlabel("弱学习器数量")
plt.ylabel("准确率 (%)")
plt.title("不同数量弱学习器AdaBoost的训练集与测试集准确率对比")
plt.legend()
plt.grid(True)
plt.savefig('./Pictures/Accuracies_AdaBoost.png')
plt.show()

# # 准备绘制基础决策树的决策边界
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                      np.arange(y_min, y_max, 0.01))
#
# # 预测网格上的每一点
# Z_tree = base_learner.predict(np.c_[xx.ravel(), yy.ravel()])
# Z_tree = Z_tree.reshape(xx.shape)
#
# # 绘制决策边界图像
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
# # 准备绘制 AdaBoost 的决策边界
# Z_ada = adaboost_model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z_ada = Z_ada.reshape(xx.shape)
#
# # 绘制决策边界
# plt.figure(figsize=(12, 7))
# plt.contourf(xx, yy, Z_ada, alpha=0.8, cmap=plt.cm.Paired)
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
# plt.xlabel("花瓣长度(cm)")
# plt.ylabel("花瓣宽度(cm)")
# plt.title(f"基础决策树分类, 训练集准确率: {train_accuracy_tree:.6f}%，测试集准确率: {test_accuracy_tree:.6f}%")
# plt.legend(loc="upper left", fontsize=12)
# plt.colorbar()  # 添加颜色条
# plt.savefig('./Pictures/DecisionTreeClassifier.png')
# plt.show()
#
# # 图例和标签
# plt.xlabel("花瓣长度(cm)")
# plt.ylabel("花瓣宽度(cm)")
# plt.title(f"AdaBoost 分类, 训练集准确率: {train_accuracy_ada:.6f}%，测试集准确率: {test_accuracy_ada:.6f}%")
# plt.legend(loc="upper left", fontsize=12)
# plt.colorbar()  # 添加颜色条
# plt.savefig('./Pictures/AdaBoost_Classification.png')
# plt.show()
