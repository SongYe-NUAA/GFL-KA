import pandas as pd

# 读取Excel文件
df = pd.read_excel(r'C:\Users\Administrator\Desktop\wind_dataset\数据集介绍.xlsx', sheet_name='model_precision')  # 如果是.xls文件，pandas同样支持

# 将DataFrame保存为CSV文件
df.to_csv('output1.csv', index=False)

print("Excel file has been converted to CSV.")