import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from tensorflow.keras.datasets import fashion_mnist
import os

# 指定输出文件夹路径
output_folder = '手工特征'
os.makedirs(output_folder, exist_ok=True)

# 加载数据
print("Loading data...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Data loaded.")

# 提取像素值特征并显示
def extract_pixel_features(images):
    print("Extracting pixel features...")
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
    for i in range(5):  # 只显示前5张图像的像素特征
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.title(f'Pixel Feature Image {i}')
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.savefig(os.path.join(output_folder, f'pixel_feature_image_{i}.png'))
        plt.close()
    return images.reshape(images.shape[0], -1)

# 提取 HOG 特征并显示
def extract_hog_features(images):
    print("Extracting HOG features...")
    hog_features = []
    for i, image in enumerate(images):
        feature, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(feature)
        if i < 5:  # 只显示前5张图像的 HOG 特征
            plt.figure(figsize=(4, 4))
            plt.axis('off')
            plt.title(f'HOG Feature Image {i}')
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            plt.imshow(hog_image_rescaled, cmap='gray')
            plt.savefig(os.path.join(output_folder, f'hog_feature_image_{i}.png'))
            plt.close()
        if i % 1000 == 0:
            print(f"Processed {i} images for HOG features")
    return np.array(hog_features)

# 提取 LBP 特征并显示
def extract_lbp_features(images):
    print("Extracting LBP features...")
    lbp_features = []
    for i, image in enumerate(images):
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        lbp_features.append(lbp.flatten())
        if i < 5:  # 只显示前5张图像的 LBP 特征
            plt.figure(figsize=(4, 4))
            plt.axis('off')
            plt.title(f'LBP Feature Image {i}')
            plt.imshow(lbp, cmap='gray')
            plt.savefig(os.path.join(output_folder, f'lbp_feature_image_{i}.png'))
            plt.close()
        if i % 1000 == 0:
            print(f"Processed {i} images for LBP features")
    return np.array(lbp_features)

# 提取所有特征
x_train_pixel = extract_pixel_features(x_train)
x_test_pixel = extract_pixel_features(x_test)
print("Pixel features extracted.")

x_train_hog = extract_hog_features(x_train)
x_test_hog = extract_hog_features(x_test)
print("HOG features extracted.")

x_train_lbp = extract_lbp_features(x_train)
x_test_lbp = extract_lbp_features(x_test)
print("LBP features extracted.")

# 保存数据
print("Saving data to .npy files...")
np.save(os.path.join(output_folder, 'x_train_pixel.npy'), x_train_pixel)
np.save(os.path.join(output_folder, 'x_test_pixel.npy'), x_test_pixel)
np.save(os.path.join(output_folder, 'x_train_hog.npy'), x_train_hog)
np.save(os.path.join(output_folder, 'x_test_hog.npy'), x_test_hog)
np.save(os.path.join(output_folder, 'x_train_lbp.npy'), x_train_lbp)
np.save(os.path.join(output_folder, 'x_test_lbp.npy'), x_test_lbp)
np.save(os.path.join(output_folder, 'y_train.npy'), y_train)
np.save(os.path.join(output_folder, 'y_test.npy'), y_test)
print("Data saved successfully.")
