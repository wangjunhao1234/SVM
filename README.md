# FashionMNIST Project

## 1. 手工特征

### 什么是手工特征？
在机器学习中，特别是在图像处理和计算机视觉领域，手工特征是指那些由研究者根据对问题的理解和经验，手动设计的特征，以帮助算法更好地理解和分析数据。这些特征通常是从原始数据中提取的，用于提供有关数据的重要信息，以便训练模型可以更有效地进行预测或分类。下面详细介绍一些常见的手工特征：

1. **像素值**：最直接的图像特征，即图像中每个像素点的颜色值。在处理深度学习之前，直接使用像素值作为特征输入到机器学习模型中是非常常见的。

2. **直方图定向梯度（Histogram of Oriented Gradients, HOG）**：HOG是一种在计算机视觉和图像处理中广泛使用的特征描述符。这种方法通过计算图像局部区域中的梯度方向直方图来捕捉图像的形状和纹理信息。HOG特征对于物体检测（特别是行人检测）非常有效。

3. **局部二值模式（Local Binary Patterns, LBP）**：LBP是一个用于纹理分类的简单而有效的纹理运算符，它通过比较每个像素与其周围邻域的像素，来生成一个二进制数。这个二进制数可以反映出局部区域的纹理信息，LBP对于面部识别等应用非常有用。

这些手工特征在深度学习流行之前非常流行，因为它们可以提供很多关于图像的有用信息，帮助传统的机器学习模型（如支持向量机和决策树）进行有效的学习。然而，在深度学习中，通常优先使用由卷积神经网络（CNN）自动学习的特征，因为这些特征可以更全面和抽象地捕捉图像中的复杂模式。

### 什么是VLFeat？
VLFeat是一个开源的计算机视觉算法库，主要用于实现各种标准的计算机视觉和机器学习算法，尤其是那些与图像识别相关的算法。它为研究和开发提供了一系列的工具，以帮助快速地实现复杂的视觉算法并进行实验。

#### 主要特点
1. **多种算法实现**：VLFeat提供多种流行的视觉算法的实现，包括但不限于SIFT（尺度不变特征变换）、HOG（直方图定向梯度）、K-means聚类、快速近似最近邻搜索、支持向量机（SVM）等。
2. **高效性能**：这个库的算法实现着重于效率和速度，适合在真实世界的应用中使用，特别是在处理大规模数据时。
3. **易于使用的API**：VLFeat提供了清晰和一致的API，方便用户调用。它支持C和MATLAB接口，使得从原型到实际应用的转化更加便捷。
4. **跨平台**：它可以在多种操作系统上编译和运行，包括Windows、Mac OS X和Linux。

#### 应用领域
VLFeat广泛用于学术研究和工业界，特别是在图像匹配、物体识别、图像分类等领域。它的算法库对于快速原型开发和实验性研究特别有用，因为它简化了复杂算法的实现过程，使研究者可以将更多的精力集中在创新和改进算法上。

#### 开源和社区支持
作为一个开源项目，VLFeat鼓励社区贡献和代码改进，这也促进了其功能的不断扩展和优化。这种开放的开发模式有助于保持库的现代性，与最新的研究和技术发展保持同步。

### 本项目采用的手工特征
本报告选用了以下手工特征，并利用VLFeat这一现有代码库提取：
1. 像素值
2. HOG（直方图定向梯度）
3. LBP（局部二值模式）

具体实现细节和代码示例请参考项目的源代码。

## 2. 项目结构

- `data/`：数据集目录
- `src/`：源代码目录
  - `features.py`：手工特征提取代码
  - `train.py`：模型训练代码
  - `evaluate.py`：模型评估代码
- `results/`：结果保存目录
- `README.md`：项目说明文档

## 3. 安装和使用

### 依赖安装
请确保已安装以下依赖：
- Python 3.6+
- TensorFlow
- VLFeat

### 数据集
请将FashionMNIST数据集下载并解压到`data/`目录下。

### 手工特征提取
运行以下命令提取手工特征：
```bash
python src/features.py
