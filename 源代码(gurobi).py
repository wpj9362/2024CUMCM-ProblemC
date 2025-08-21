import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

# 初始化矩阵        
rows, cols = 55, 42
origin_matrix = [[0] * cols for _ in range(rows)]
s = [row.copy() for row in origin_matrix]  # 确保深拷贝
c = [row.copy() for row in origin_matrix]
n = [row.copy() for row in origin_matrix]
a = [0]*55

# 定义数据填充规则：(起始行, 数据范围, 目标列范围)
fill_rules = [
    (1, slice(0, 15), slice(1,16)),   # s[1][1:16] = data[0:15]
    (7, slice(15, 30), slice(1,16)),  # s[7][1:16] = data[15:30]
    (21, slice(30, 45), slice(1,16)), # s[21][1:16] = data[30:45]
    (27, slice(45, 46), slice(16,17)), # s[27][16] = data[45]
    (27, slice(46, 64), slice(17,35)), # s[27][17:35] = data[46:64]
    (35, slice(64, 85), slice(17,38)), # s[35][17:38] = data[64:85]
    (35, slice(85, 89), slice(38,42)), # s[35][38:42] = data[85:89]
    (51, slice(89, 107), slice(17,35)), # s[51][17:35] = data[89:107]
]
# 定义行复制规则：(起始行, 结束行, 复制源行)
copy_rules = [
    (2, 7, 1),    # s[2:7] = s[1].copy()
    (8, 21, 7),   # s[8:21] = s[7].copy()
    (22, 27, 21), # s[22:27] = s[21].copy()
    (28, 35, 27), # s[28:35] = s[27].copy()
    (36, 51, 35), # s[36:51] = s[35].copy()
    (52, 55, 51), # s[52:55] = s[51].copy()
]
# 读取Excel数据
df1 = pd.read_excel("附件2.xlsx", sheet_name="2023年统计的相关数据")
sales = df1.iloc[0:107,8].tolist()  # 将售价转换为列表
costs = df1.iloc[0:107,6].tolist()   # 每亩的成本
nums = df1.iloc[0:107,5] #亩产量
df2 = pd.read_excel("附件1.xlsx",sheet_name="乡村的现有耕地")
areas = df2.iloc[0:54,2].tolist()
for number,area in enumerate(areas,start=1):
    a[number]=area

#填入之前的数据                         
for target_row, data_slice, col_slice in fill_rules:
    s[target_row][col_slice] = sales[data_slice]
    c[target_row][col_slice] = costs[data_slice]
    n[target_row][col_slice] = nums[data_slice]

# 按规则复制行
for start_row, end_row, source_row in copy_rules:
    for row in range(start_row, end_row):
        s[row] = s[source_row].copy()
        c[row] = c[source_row].copy()
        n[row] = n[source_row].copy()

# 创建Gurobi模型
model = gp.Model("农作物的种植策略")
model.ModelSense = GRB.MAXIMIZE  # 设置为最大化问题

years = range(0,8)
fields = range(1,55) 
crops = range(1,42)  
seasons = [0,1]  

# 定义变量
x = {}  # 种植面积变量
z = {}  # 二进制变量，表示是否有种植
y = {}  # 二进制变量，用于D地水稻模式

# 创建变量
for t in years:
    for ss in seasons:
        for i in fields:
            for j in crops:
                # x变量：整数变量，范围0-2
                x[t, ss, i, j] = model.addVar(lb=0, ub=2, vtype=GRB.INTEGER, 
                                             name=f"x_{t}_{ss}_{i}_{j}")
                # z变量：二进制变量，与x关联
                z[t, ss, i, j] = model.addVar(vtype=GRB.BINARY, 
                                             name=f"z_{t}_{ss}_{i}_{j}")

# y变量：用于D地水稻模式
for t in years:
    for i in fields:
        y[t, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}_{i}")

# n_loss_nonneg变量：非负损失变量
n_loss_nonneg = {}
for t in years:
    for num in [0, 1, 2, 3]:
        for ss in seasons:
            for j in crops:
                n_loss_nonneg[t, num, ss, j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                           name=f"n_loss_{t}_{num}_{ss}_{j}")

# 更新模型以添加变量
model.update()

# 添加x和z之间的关联约束
for t in years:
    for ss in seasons:
        for i in fields:
            for j in crops:
                # x>0 则 z 必须为1
                model.addConstr(x[t, ss, i, j] <= 2 * z[t, ss, i, j], 
                               f"link1_{t}_{ss}_{i}_{j}")
                # z=1 则 x 至少为1
                model.addConstr(x[t, ss, i, j] >= z[t, ss, i, j], 
                               f"link2_{t}_{ss}_{i}_{j}")

# 读取已知数据并设置t=0的约束
df3 = pd.read_excel("附件2.xlsx", sheet_name="2023年的农作物种植情况",  
                   usecols=["ss", "i", "j", "取值"])
known_data = df3.copy()
known_dict = {}
for idx, row in known_data.iterrows():
    ss = row["ss"]
    i = row["i"]
    j = row["j"]
    value = row["取值"]
    known_dict[(ss, i, j)] = value

# 为t=0设置固定值
for t in years:
    if t==0:
        for ss in seasons:
            for i in fields:
                for j in crops:
                    default_value = 0
                    if (ss, i, j) in known_dict:
                        default_value = known_dict[(ss, i, j)]
                    model.addConstr(x[t, ss, i, j] == default_value, 
                                   f"fixed_{t}_{ss}_{i}_{j}")
                    
# 约束条件
# (1) 所有地必须种满
for t in range(1,8):
    for i in range(1,27):   # A,B,C地
        model.addConstr(quicksum(x[t, 0, i, j] for j in range(1,16)) == 2,
                       f"ABC_season0_{t}_{i}")
        model.addConstr(quicksum(x[t, 1, i, j] for j in range(1,16)) == 0,
                       f"ABC_season1_{t}_{i}")
    
    for i in range(27,35):  # D地
        model.addConstr(0.5 * x[t, 0, i, 16] == y[t, i], f"D_rice_{t}_{i}")
        model.addConstr(quicksum(0.5 * x[t, 0, i, j] for j in range(17,35)) == 1 - y[t, i],
                       f"D_dry_season0_{t}_{i}")
        model.addConstr(quicksum(0.5 * x[t, 0, i, j] for j in range(35,38)) == 0,
                       f"D_zero1_{t}_{i}")
        model.addConstr(quicksum(0.5 * x[t, 1, i, j] for j in range(17,35)) == 0,
                       f"D_zero2_{t}_{i}")
        model.addConstr(quicksum(0.5 * x[t, 1, i, j] for j in range(35,38)) == 1 - y[t, i],
                       f"D_wet_season1_{t}_{i}")
    
    for i in range(35,51):  # E地
        model.addConstr(quicksum(x[t, 0, i, j] for j in range(17,35)) == 2,
                       f"E_season0_{t}_{i}")
        model.addConstr(quicksum(x[t, 0, i, j] for j in range(35,42)) == 0,
                       f"E_zero1_{t}_{i}")
        model.addConstr(quicksum(x[t, 1, i, j] for j in range(17,38)) == 0,
                       f"E_zero2_{t}_{i}")
        model.addConstr(quicksum(x[t, 1, i, j] for j in range(38,42)) == 2,
                       f"E_season1_{t}_{i}")
    
    for i in range(51,55):  # F地
        model.addConstr(quicksum(x[t, 0, i, j] for j in range(17,35)) == 2,
                       f"F_season0_{t}_{i}")
        model.addConstr(quicksum(x[t, 1, i, j] for j in range(17,35)) == 2,
                       f"F_season1_{t}_{i}")
        model.addConstr(quicksum(x[t, ss, i, j] for ss in seasons for j in range(35,42)) == 0,
                       f"F_zero_{t}_{i}")
        for j in range(17,35):
            model.addConstr(z[t, 0, i, j] + z[t, 1, i, j] <= 1,
                           f"F_no_repeat_{t}_{i}_{j}")

# (2) 保证不重复种植
valid_crops = [[] for _ in range(55)]
for i in range(55):
    valid_crops[i] = [j for j in crops if s[i][j] != 0]

for t in range(0,7):
    for i in fields:
        for j in valid_crops[i]:
            model.addConstr(quicksum(z[t, ss, i, j] + z[t+1, ss, i, j] for ss in seasons) <= 1,
                           f"no_repeat_{t}_{i}_{j}")

# (3) 保证每三年至少种一次豆类
for t in range(0,6):
    for i in range(1,27):
        model.addConstr(quicksum(z[t, 0, i, j] + z[t+1, 0, i, j] + z[t+2, 0, i, j] 
                              for j in range(1,6)) >= 1, f"beans_ABC_{t}_{i}")
    for i in range(27,55):
        model.addConstr(quicksum(z[t, ss, i, j] + z[t+1, ss, i, j] + z[t+2, ss, i, j] 
                              for ss in seasons for j in range(17,20)) >= 1, f"beans_DEF_{t}_{i}")

# (4) 限制每种作物每年的种植数量
for t in range(1,8):
    for j in crops:
        model.addConstr(quicksum(z[t, ss, i, j] for ss in seasons for i in fields) <= 5,
                       f"crop_limit_{t}_{j}")
        

# 定义订单作物和地块
order_crops = {
    (0,0): range(1,16),
    (0,1): [],
    (1,0): range(16,35),
    (1,1): range(35,38),
    (2,0): range(16,35),
    (2,1): range(38,42),
    (3,0): range(16,35),
    (3,1): range(16,35)
}

order_fields = {
    0: range(1,27),
    1: range(27,35),
    2: range(35,51),
    3: range(51,55)  # 修正为55以包含所有地块
}

order_prices={
    0:1,
    1:27,
    2:35,
    3:51
}

# 计算损失约束
for t in range(1,8):
    for num in range(4):
        for ss in seasons:    #   x[0,0,4,1] = 2    
            if (num, ss) in order_crops:
                for j in order_crops[(num, ss)]:
                    # 计算n_loss
                    left_expr=quicksum(0.5 * x[t, ss, i, j] * n[i][j]*a[i] for i in order_fields[num])
                    right_expr=quicksum(0.5 * x[0, ss, i, j] * n[i][j]*a[i] for i in order_fields[num])
                    n_loss_expr = left_expr - right_expr
                    # 添加约束： 相比2023年种植任一作物的量不要太低
                    model.addConstr(left_expr >= 0.5*right_expr)
                    # 添加非负约束
                    model.addConstr(n_loss_nonneg[t, num, ss, j] >= n_loss_expr,
                                   f"n_loss_lb_{t}_{num}_{ss}_{j}")
                    model.addConstr(n_loss_nonneg[t, num, ss, j] >= 0,
                                   f"n_loss_nonneg_{t}_{num}_{ss}_{j}")

# 目标函数
w_expected = {}
w_loss = {}
w = {}

for t in years:
    # 计算期望收益
    w_expected[t] = quicksum(
        0.5 * x[t, ss, i, j] *a[i] *(s[i][j]*n[i][j]-c[i][j])  #
        for ss in seasons
        for i in fields
        for j in crops
    )

# 初始化第0年的损失为0
w_loss[0] = 0
loss_terms = []
for t in range(1,8):
    for num in range(0,4):
        for ss in seasons:
            if (num, ss) in order_crops:
                for j in order_crops[(num, ss)]:      
                    loss_terms.append(n_loss_nonneg[t, num, ss, j] * s[order_prices[num]][j])
    
    w_loss[t] = quicksum(loss_terms)

# 计算各年的总收益
for t in years:
    if t == 0:
        w[t] = w_expected[t]
    else:
        w[t] = w_expected[t] - w_loss[t]

# 设置总目标函数
model.setObjective(quicksum(w[t] for t in years))
# 设置求解参数                                                                                                                
model.setParam('TimeLimit', 3600)    # 1小时时间限制 
model.setParam('Threads', 4)         # 使用4个线程 
model.setParam('MIPGap', 0.006)      # 1%的最优间隙 
model.setParam('OutputFlag', 1)      # 显示求解过程               
                                                      
# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"找到最优解，目标值: {model.ObjVal}")
elif model.status == GRB.TIME_LIMIT:
    print(f"达到时间限制，当前最佳目标值: {model.ObjVal}")
else:
    print(f"求解状态: {model.status}")

# 输出非零变量
print("\n非零变量取值:")
for v in model.getVars():
    if v.X >=1 :  # 忽略接近0的值
        print(f"{v.VarName}: {v.X}")

# 输出求解统计信息
print(f"\n求解时间: {model.Runtime:.2f} 秒")
print(f"\n最优间隙: {model.MIPGap:.4%}")
print(f"节点数: {model.NodeCount}")

