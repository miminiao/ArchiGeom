import numpy as np

# 定义损失函数
def loss_function(theta, X, y):
    # theta: 模型参数，X: 特征矩阵，y: 标签向量
    m = len(y)
    h = sigmoid(np.dot(X, theta))  # 假设函数
    J = -1/m * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))  # 二分类问题的对数似然损失函数
    return J

# 计算假设函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 计算梯度
def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = 1/m * np.dot(X.T, (h - y))
    return grad

# 梯度下降优化
def gradient_descent(X, y, alpha=0.01, max_iter=1000, epsilon=1e-5):
    # X: 特征矩阵，y: 标签向量，alpha: 学习率，max_iter: 最大迭代次数，epsilon: 收敛精度
    m, n = X.shape
    theta = np.zeros((n, 1))  # 初始化模型参数
    iter_count = 0
    while iter_count < max_iter:
        J = loss_function(theta, X, y)
        grad = gradient(theta, X, y)
        theta_new = theta - alpha * grad  # 更新参数
        if np.linalg.norm(theta_new - theta) < epsilon:  # 判断是否收敛
            break
        iter_count += 1
        theta = theta_new
    return theta

