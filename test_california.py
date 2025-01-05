# 导入所需库
from sklearn.datasets import fetch_california_housing  # 加载加州房价数据集
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import r2_score, mean_squared_error  # 评估指标
import matplotlib.pyplot as plt  # 导入绘图库

plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片标题中文显示
plt.rcParams['axes.unicode_minus'] = False

# 加载加州房价数据集
data = fetch_california_housing()  # 数据加载
X = data.data  # 特征变量
y = data.target  # 目标变量（房价）

# 划分训练集和测试集，测试集比例设为25%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=420
)

# 初始化线性回归模型
model = LinearRegression()

# 使用训练集训练模型
model.fit(X_train, y_train)

# 使用训练好的模型对测试集进行预测
y_pred = model.predict(X_test)

# 对模型进行评估
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
r2 = r2_score(y_test, y_pred)  # 计算R^2

# 输出结果
print(f"线性回归的均方误差: {mse:.6f}")
print(f"线性回归的R^2: {r2:.6f}")

# 绘制实际值与预测值的对比图
plt.figure(figsize=(10, 6))  # 设置图像大小

# 绘制散点图：真实房价 vs 预测房价
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='预测值')

# 绘制参考线：完全拟合的线
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='完全拟合线')

# 设置图例、标题和坐标轴标签
plt.xlabel('实际房价', fontsize=12)  # X轴标签
plt.ylabel('预测房价', fontsize=12)  # Y轴标签
plt.title('实际房价与预测房价的对比', fontsize=14)  # 图标题
plt.legend(fontsize=10)  # 图例
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格方便观察

# 保存图像
plt.savefig('./Pictures/Linear Regression.png')
