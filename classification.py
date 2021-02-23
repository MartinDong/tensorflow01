# Tensorflow and tf.keras
# Helper libraries
# 支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
# 画图的函数
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# 导入[FashionMNIST]数据集
fashion_mnist = keras.datasets.fashion_mnist
# 装载数据
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 每个图像都会被映射到一个标签。由于数据集不包括类名称，请将它们存储在下方，供稍后绘制图像时使用：
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 浏览数据
## 在训练模型之前，我们先浏览一下数据集的格式。以下代码显示训练集中有 60,000 个图像，每个图像由 28 x 28 的像素表示：
print(train_images.shape)

## 查看训练集大小
print(len(train_labels))

## 每个标签都是一个 0 到 9 之间的整数：
print(train_labels)

## 测试集中有 10,000 个图像。同样，每个图像都由 28x28 个像素表示：
print(test_images.shape)

## 测试集包含 10,000 个图像标签：
print(len(test_labels))

# 预处理数据
## 在训练网络之前，必须对数据进行预处理。如果您检查训练集中的第一个图像，您会看到像素值处于 0 到 255 之间：
# plt.figure()
# # 要展示的图片
# plt.imshow(train_images[1])
# plt.xlabel(class_names[train_labels[1]])
# # 颜色条
# plt.colorbar()
# # 是否显示网格
# plt.grid(True)
# # 展示
# plt.show()

## 将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型。为此，请将这些值除以 255。请务必以相同的方式对训练集和测试集进行预处理：
train_images = train_images / 255.0
test_images = test_images / 255.0

## 为了验证数据的格式是否正确，以及您是否已准备好构建和训练网络，让我们显示训练集中的前 25 个图像，并在每个图像下方显示类名称。
plt.figure(figsize=(10, 10))
for i in range(10):
    # 创建单个子图，5X5的矩阵
    plt.subplot(5, 5, i + 1)
    # 设置横轴记号
    plt.xticks([])
    # 设置纵轴记号
    plt.yticks([])
    # 是否显示网格
    plt.grid(False)
    # 设置要展示的图片
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 设置横轴的标签
    plt.xlabel(class_names[train_labels[i]])
    # 设置纵轴标签
    plt.ylabel(class_names[train_labels[i]])

plt.show()

# 构建模型
## 构建神经网络需要先配置模型的层，然后再编译模型。
## 设置层
## 神经网络的基本组成部分是层。层会从向其馈送的数据中提取表示形式。希望这些表示形式有助于解决手头上的问题。
## 大多数深度学习都包括将简单的层链接在一起。大多数层（如 tf.keras.layers.Dense）都具有在训练期间才会学习的参数。
model = keras.Sequential([
    # 将一个维度大于或等于3的高维矩阵，“压扁”为一个二维矩阵。即保留第一个维度（如：batch的个数），然后将剩下维度的值相乘作为“压扁”矩阵的第二个维度。
    # tf.keras.layers.Flatten 将图像格式从二维数组（28 x 28 像素）转换成一维数组（28 x 28 = 784 像素）
    # 将该层视为图像中未堆叠的像素行并将其排列起来。该层没有要学习的参数，它只会重新格式化数据。
    keras.layers.Flatten(input_shape=(28, 28)),
    # 展平像素后，网络会包括两个 tf.keras.layers.Dense 层的序列。它们是密集连接或全连接神经层。
    # 第一个 Dense 层有 128 个节点（或神经元）。
    keras.layers.Dense(128, activation='relu'),
    # 第二个（也是最后一个）层会返回一个长度为 10 的 logits 数组。每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。
    keras.layers.Dense(10)
])

# 编译模型
# 在准备对模型进行训练之前，还需要再对其进行一些设置。以下内容是在模型的编译步骤中添加的：
# - 损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
# - 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# - 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(
    optimizer='adam',
    # 分类：损失函数
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Accuracy（准确率）是机器学习中最简单的一种评价模型好坏的指标
    metrics=['accuracy']
)
