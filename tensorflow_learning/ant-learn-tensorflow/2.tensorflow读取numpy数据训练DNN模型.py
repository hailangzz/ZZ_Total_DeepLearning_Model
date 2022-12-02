import numpy as np
import tensorflow as tf

# numpy文件地址
filename = "D:/PycharmProgram/datas/mnist/mnist.npz"

with np.load(filename) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']
print(type(train_examples), type(train_labels))
print(train_examples.ndim, train_labels.ndim)
print(train_examples.shape, train_labels.shape)
print(train_examples.dtype, train_labels.dtype)

print(train_examples[0])
print(train_examples[0].shape)
print(train_labels[0])

# import matplotlib.pyplot as plt
# plt.imshow(train_examples[0])
# plt.show()

# 加载数据
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

# 查看数据集的一个样本（这时包含了所有特征列、标签列）
print(train_dataset.as_numpy_iterator().next()[0].shape,train_dataset.as_numpy_iterator().next())

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

shuffle_ds = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)

train_dataset = shuffle_ds.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
# input_shape要省略输入数据的第一维度，(60000, 28, 28)，只需要输入(28,28)
# 这里的input_shape其实就等于train_examples.shape[1:]
first_layer = tf.keras.layers.Flatten(input_shape=(28, 28))

# 搭建模型
model = tf.keras.Sequential([
    first_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 模型编译
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),  #当输出为稀疏交叉熵分类时使用此损失函数;
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# 查看模型信息
print(model.summary())
model.fit(train_dataset, epochs=10)
model.evaluate(test_dataset)
model.save(r'D:/PycharmProgram/total_model/my_model2.h5')





























