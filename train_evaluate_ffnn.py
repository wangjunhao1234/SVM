import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import pandas as pd
import os

# 指定输出文件夹路径
ffnn_output_folder = 'FFNN训练与评估'
os.makedirs(ffnn_output_folder, exist_ok=True)

# 加载数据
print("Loading data...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Data loaded.")

# 数据预处理
print("Normalizing data...")
x_train = x_train / 255.0
x_test = x_test / 255.0
print("Data normalized.")

# 构建前馈神经网络
print("Building the feedforward neural network...")
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
print("Model built.")

# 编译模型
print("Compiling the model...")
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled.")

# 训练模型
print("Training the model...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
print("Model training completed.")

# 评估模型
print("Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# 保存模型
print("Saving the model...")
model.save(os.path.join(ffnn_output_folder, 'ffnn_model.h5'))
print("Model saved as 'ffnn_model.h5'.")

# 保存训练过程数据
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

# 创建 DataFrame 保存训练和验证准确率
df_ffnn = pd.DataFrame({
    'Epoch': epochs,
    'Train Accuracy': train_acc,
    'Validation Accuracy': val_acc
})

# 保存为 CSV 文件
df_ffnn.to_csv(os.path.join(ffnn_output_folder, 'ffnn_training_results.csv'), index=False)
print("Training results saved as 'ffnn_training_results.csv'.")

# 绘制 FFNN 的训练和验证准确率图表
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('FFNN Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(ffnn_output_folder, 'ffnn_accuracy.png'))
plt.show()
print("FFNN accuracy chart saved as 'ffnn_accuracy.png'.")
