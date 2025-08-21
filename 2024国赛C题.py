# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import seaborn as sns

# 设置中文显示及负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示

# 1. 数据读取
def read_data():
    # 读取作物信息和地块信息，假设数据存储在CSV文件中
    crops = pd.read_csv('crops_data.csv')  # 包含作物的亩产量、销售单价、成本等
    fields = pd.read_csv('fields_data.csv')  # 包含每块地块的面积
    return crops, fields

# 2. 线性规划模型构建
def build_model(crops, fields):
    # 提取作物数量和地块数量
    M = len(crops)  # 作物数量
    N = len(fields)  # 地块数量
    
    # 2.1 计算每种作物的收益
    crops['收益'] = crops['亩产量'] * crops['销售单价'] - crops['种植成本']

    # 2.2 设置目标函数系数（为便于优化，取负数）
    c = -crops['收益'].values  # 目标为收益最大化，因此用负值求解最小化问题

    # 2.3 设置约束条件
    # 约束1：每块地块的种植面积不超过其面积
    A_ub = np.zeros((N, M))  # 不等式约束矩阵
    b_ub = fields['面积'].values  # 每块地块的面积

    # 约束赋值：每块地块的作物种植面积之和小于等于该地块面积
    for i in range(N):
        for j in range(M):
            A_ub[i][j] = 1  # 允许每种作物在每块地块上种植
    
    # 约束2：非负性约束
    bounds = [(0, None) for _ in range(M)]  # 每种作物的种植面积必须非负

    # 2.4 轮作约束：豆类作物在任意三年内至少要种植一次
    # 这里假设豆类作物的索引为0
    # T 表示种植季数（假设为2季）
    T = 2
    A_eq = []
    b_eq = []
    for n in range(1, 4):  # n为年份标记
        eq_row = np.zeros(M)
        eq_row[0] = -1  # 假设豆类作物索引是0
        A_eq.append(eq_row)
        b_eq.append(-1)  # 至少种植一次的约束

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # 2.5 合并约束
    return c, A_ub, b_ub, A_eq, b_eq, bounds

# 3. 求解模型
def solve_model(c, A_ub, b_ub, A_eq, b_eq, bounds):
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        print("优化成功！")
    else:
        print("优化失败！", res.message)
    
    return res

# 4. 结果分析与可视化
def visualize_results(res, crops):
    # 获取优化后的种植面积
    planted_area = res.x
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        '作物': crops['作物名称'],
        '种植面积': planted_area
    })

    # 打印结果
    print("种植结果：")
    print(results)

    # 绘制结果的柱状图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='作物', y='种植面积', data=results, palette='Blues_d')
    plt.title('各作物种植面积分配')
    plt.xlabel('作物')
    plt.ylabel('种植面积（亩）')
    plt.xticks(rotation=45)
    plt.savefig('crop_distribution.png')  # 保存结果图
    plt.show()  # 显示结果图

# 5. 主函数
def main():
    # 数据读取
    crops, fields = read_data()
    
    # 构建模型
    c, A_ub, b_ub, A_eq, b_eq, bounds = build_model(crops, fields)
    
    # 求解模型
    res = solve_model(c, A_ub, b_ub, A_eq, b_eq, bounds)
    
    # 结果可视化
    visualize_results(res, crops)

# 执行主函数
if __name__ == '__main__':
    main()