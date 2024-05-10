import numpy as np
import pandas as pd
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import os
import time

# 指定输入和输出文件夹路径
input_folder = '手工特征'
output_folder = 'SVM训练和评估'
os.makedirs(output_folder, exist_ok=True)

# 加载数据
print("Loading data...")
x_train_pixel = np.load(os.path.join(input_folder, 'x_train_pixel.npy'))
x_test_pixel = np.load(os.path.join(input_folder, 'x_test_pixel.npy'))
x_train_hog = np.load(os.path.join(input_folder, 'x_train_hog.npy'))
x_test_hog = np.load(os.path.join(input_folder, 'x_test_hog.npy'))
x_train_lbp = np.load(os.path.join(input_folder, 'x_train_lbp.npy'))
x_test_lbp = np.load(os.path.join(input_folder, 'x_test_lbp.npy'))
y_train = np.load(os.path.join(input_folder, 'y_train.npy'))
y_test = np.load(os.path.join(input_folder, 'y_test.npy'))
print("Data loaded.")

# 检查数据集大小
print(f"x_train_pixel shape: {x_train_pixel.shape}")
print(f"x_test_pixel shape: {x_test_pixel.shape}")
print(f"x_train_hog shape: {x_train_hog.shape}")
print(f"x_test_hog shape: {x_test_hog.shape}")
print(f"x_train_lbp shape: {x_train_lbp.shape}")
print(f"x_test_lbp shape: {x_test_lbp.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 定义函数训练和评估 SVM
def train_and_evaluate_svm(x_train, y_train, x_test, y_test, feature_name, kernel='linear'):
    print(f"Training SVM model with {feature_name} features and {kernel} kernel...")
    model = svm.SVC(kernel=kernel)

    # 添加训练进度日志
    n_samples = len(x_train)
    chunk_size = n_samples // 10  # 训练进度日志的粒度，这里设为10%
    for i in range(0, n_samples, chunk_size):
        end = i + chunk_size if i + chunk_size < n_samples else n_samples
        print(f"Training on samples {i} to {end} out of {n_samples}...")
        model.fit(x_train[i:end], y_train[i:end])
        print(f"Trained on samples {i} to {end}.")

    print(f"SVM model with {feature_name} features and {kernel} kernel trained.")

    print(f"Making predictions on test data with {feature_name} features and {kernel} kernel...")
    y_pred = model.predict(x_test)
    print(f"Predictions with {feature_name} features and {kernel} kernel completed.")

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"{feature_name} Features with {kernel} Kernel Accuracy: {accuracy}")

    # 绘制混淆矩阵
    print(f"Plotting confusion matrix for {feature_name} features and {kernel} kernel...")
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(output_folder, f'confusion_matrix_{feature_name}_{kernel}.png'))
    plt.close()
    print(f"Confusion matrix for {feature_name} features and {kernel} kernel saved as 'confusion_matrix_{feature_name}_{kernel}.png'.")
    return accuracy

# 训练和评估 SVM 模型
features_kernels = [
    ('pixel', 'linear'),
    ('pixel', 'rbf'),
    ('pixel', 'poly'),
    ('hog', 'linear'),
    ('hog', 'rbf'),
    ('hog', 'poly'),
    ('lbp', 'linear'),
    ('lbp', 'rbf'),
    ('lbp', 'poly')
]

accuracies = []

print("Starting SVM training and evaluation...")
for feature, kernel in features_kernels:
    print(f"Processing feature: {feature}, kernel: {kernel}")
    if feature == 'pixel':
        x_train, x_test = x_train_pixel, x_test_pixel
    elif feature == 'hog':
        x_train, x_test = x_train_hog, x_test_hog
    elif feature == 'lbp':
        x_train, x_test = x_train_lbp, x_test_lbp
    start_time = time.time()
    accuracy = train_and_evaluate_svm(x_train, y_train, x_test, y_test, feature, kernel)
    elapsed_time = time.time() - start_time
    accuracies.append((feature, kernel, accuracy, elapsed_time))
    print(f"Completed feature: {feature}, kernel: {kernel} with accuracy: {accuracy} in {elapsed_time:.2f} seconds.")

# 打印所有结果
print("All SVM training and evaluation completed. Here are the results:")
for feature, kernel, accuracy, elapsed_time in accuracies:
    print(f"{feature} Features with {kernel} Kernel Accuracy: {accuracy} in {elapsed_time:.2f} seconds")

# 创建 DataFrame 并保存
df_svm = pd.DataFrame(accuracies, columns=['Feature', 'Kernel', 'Accuracy', 'Elapsed Time'])
df_svm.to_csv(os.path.join(output_folder, 'svm_results.csv'), index=False)
print("SVM results saved as 'svm_results.csv'.")

# 绘制图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 SVM 结果
for feature in df_svm['Feature'].unique():
    feature_df = df_svm[df_svm['Feature'] == feature]
    ax.plot(feature_df['Kernel'], feature_df['Accuracy'], marker='o', label=f'SVM ({feature})')

# 图表设置
ax.set_xlabel('Kernel Function')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of SVM Kernel Functions')
ax.legend()
ax.grid(True)

# 保存图像
plt.savefig(os.path.join(output_folder, 'svm_kernel_comparison.png'))
plt.show()
