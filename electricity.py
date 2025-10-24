"""
electricity是一个用于部分电学实验独有的数据处理的模块，包含了一些专门用于电学实验的数据分析和计算函数。
"""
import numpy as np

def Tolerance_Of_Resistance_Box(resistor:float, min_step:float, Alpha:np.array):
    """
    计算某电阻箱测量某电阻的容差

    参数
    ----
    resistor : float
        被测电阻值(单位: 欧姆)
    min_step : float
        电阻箱最小电阻度盘的档位(单位: 欧姆)
    Alpha : array-like
        电阻箱各电阻度盘的准确度等级，从大档位到小档位，没有的档位填0，讲道理应该都有数字(单位: %)

    返回
    ----
    Tolerance : float
        电阻箱测量该电阻的容差(单位: 欧姆)
    """
    tolerance = 0
    step = min_step
    for _ in range(len(Alpha)-1):
        step *= 10
    resistor_in_step = ((int)(resistor / step) % 10) * step
    for alpha in Alpha:
        if step < min_step:
            print("Error: step < min_step")
        resistor_in_step = ((int)(resistor / step) % 10) * step
        if alpha != 0:
            tolerance += resistor_in_step * (alpha / 100)
        step /= 10
    return tolerance