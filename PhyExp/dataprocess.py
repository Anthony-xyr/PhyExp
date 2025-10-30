"""
dataprocess是一个用于处理物理实验数据的Python模块，
主要功能包括计算A类不确定度、平均值、标准差、线性拟合以及绘制数据图表。
该模块依赖于NumPy和Matplotlib库，适用于需要进行数据分析和可视化的物理实验场景。
"""
import numpy as np
import matplotlib.pyplot as plt

def calculate_A_uncertainty(data):
    """
    计算均值的A类不确定度
    参数:
        data: 数值数组
    返回:
        A类不确定度
    """
    n = len(data)
    if n < 2:
        raise ValueError("数据点数量必须大于1以计算A类不确定度")
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # 使用样本标准差
    A_uncertainty = std_dev / np.sqrt(n)
    return A_uncertainty

def calculate_uncertainty(data, delta):
    """
    综合考虑A类不确定度和B类不确定度
    计算总不确定度
    delta为仪器最大容差
    """
    A_uncertainty = calculate_A_uncertainty(data)
    B_uncertainty = delta / np.sqrt(3)  # 假设B类不确定度服从均匀分布
    total_uncertainty = np.sqrt(A_uncertainty**2 + B_uncertainty**2)
    return A_uncertainty, B_uncertainty, total_uncertainty

def mean_std(data:np.array, delta:float):
    """
    计算均值与综合不确定度

    参数
    ----
    data : array-like
        测量数据
    delta : float
        仪器最大允许误差(绝对值)

    返回
    ----
    mean : float
        数据平均值
    A_uncertainty : float
        A 类不确定度
    B_uncertainty : float
        B 类不确定度
    uncertainty : float
        综合不确定度
    """
    mean = np.mean(data)
    A_uncertainty, B_uncertainty, uncertainty = calculate_uncertainty(data, delta)
    return mean, A_uncertainty, B_uncertainty, uncertainty

def linear_fit(x, y, y_delta):
    """
    线性拟合

    参数
    ----
    x : np.array
        自变量数组
    y : np.array
        因变量数组
    y_delta : float
        因变量的仪器最大允许误差(绝对值)

    返回
    ----
    b : float
        斜率
    a : float
        截距
    r : float
        相关系数
    a_UA : float
        斜率A类不确定度
    b_UA : float
        截距A类不确定度
    a_UB : float
        斜率B类不确定度
    b_UB : float
        截距B类不确定度
    a_U : float
        斜率综合不确定度
    b_U : float
        截距综合不确定度
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x和y数组长度必须相等")
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy = x * y
    xx = x * x
    yy = y * y
    xy_mean = np.mean(xy)
    xx_mean = np.mean(xx)
    yy_mean = np.mean(yy)
    y_UB = y_delta / np.sqrt(3)

    b = (xy_mean - x_mean * y_mean) / (xx_mean - x_mean**2)
    a = y_mean - b * x_mean
    r = (xy_mean - x_mean * y_mean) / np.sqrt((xx_mean - x_mean**2) * (yy_mean - y_mean**2))

    # 计算不确定度
    b_UA = b*np.sqrt((1/r**2 - 1)/(n - 2))
    a_UA = b_UA * np.sqrt(xx_mean)
    b_UB = y_UB / np.sqrt(n*(xx_mean - x_mean**2))
    a_UB = b_UB * np.sqrt(xx_mean) 
    b_U = np.sqrt(b_UA**2 + b_UB**2)
    a_U = np.sqrt(a_UA**2 + a_UB**2)
    return b, a, r, a_UA, b_UA, a_UB, b_UB, a_U, b_U