import pandas as pd

# 假设表格数据存储在一个CSV文件中
data = pd.read_csv(r'D:\project\WTGFL\mmdetection-main\tools\output1.csv')

# 将每一列转换为向量（列表）
column_vectors = {col: data[col].tolist() for col in data.columns}

# 打印每一列的向量
for column, vector in column_vectors.items():
    print(f"{column}: {vector}")