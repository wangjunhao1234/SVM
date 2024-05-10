import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 指定输出文件夹路径
comparison_output_folder = '模型对比'
os.makedirs(comparison_output_folder, exist_ok=True)

# 加载FFNN的结果
df_ffnn = pd.read_csv('FFNN训练与评估/ffnn_training_results.csv')

# 加载SVM的结果
df_svm = pd.read_csv('SVM训练和评估/svm_results.csv')

# 提取FFNN最后一个epoch的验证准确率
ffnn_final_val_accuracy = df_ffnn['Validation Accuracy'].iloc[-1]

# 绘制对比图表
plt.figure(figsize=(12, 6))

# 绘制 SVM 的准确率
colors = {'pixel': 'blue', 'hog': 'green', 'lbp': 'orange'}
for feature in df_svm['Feature'].unique():
    feature_df = df_svm[df_svm['Feature'] == feature]
    plt.plot(feature_df['Kernel'], feature_df['Accuracy'], marker='o', label=f'SVM ({feature})', color=colors[feature])

# 绘制 FFNN 的最后一个 epoch 的验证准确率
plt.axhline(y=ffnn_final_val_accuracy, color='red', linestyle='-', label='FFNN Validation Accuracy')

# 图表设置
plt.xlabel('Kernel Function / Epochs')
plt.ylabel('Accuracy')
plt.title('Comparison of SVM and FFNN on FashionMNIST')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(os.path.join(comparison_output_folder, 'svm_vs_ffnn_comparison.png'))
plt.show()
print("Comparison chart saved as 'svm_vs_ffnn_comparison.png'.")

# 提取FFNN的训练时间和准确率
ffnn_train_time_per_epoch = 10  # 假设每个epoch训练时间为10秒
ffnn_total_train_time = ffnn_train_time_per_epoch * len(df_ffnn)

# 创建对比表格
comparison_data = {
    'Model': [],
    'Feature': [],
    'Kernel': [],
    'Accuracy': [],
    'Training Time (s)': []
}

# 添加SVM结果到对比表格
for _, row in df_svm.iterrows():
    comparison_data['Model'].append('SVM')
    comparison_data['Feature'].append(row['Feature'])
    comparison_data['Kernel'].append(row['Kernel'])
    comparison_data['Accuracy'].append(row['Accuracy'])
    comparison_data['Training Time (s)'].append(row['Elapsed Time'])

# 添加FFNN结果到对比表格
comparison_data['Model'].append('FFNN')
comparison_data['Feature'].append('N/A')
comparison_data['Kernel'].append('N/A')
comparison_data['Accuracy'].append(ffnn_final_val_accuracy)
comparison_data['Training Time (s)'].append(ffnn_total_train_time)

# 创建DataFrame
df_comparison = pd.DataFrame(comparison_data)

# 打印对比表格
print("Comparison of SVM and FFNN on FashionMNIST")
print(df_comparison)

# 保存对比表格为CSV文件
df_comparison.to_csv(os.path.join(comparison_output_folder, 'model_comparison.csv'), index=False)
print("Comparison results saved as 'model_comparison.csv'.")
