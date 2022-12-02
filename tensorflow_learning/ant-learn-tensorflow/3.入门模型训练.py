import tensorflow as tf

print(tf.__version__)
mnist = tf.keras.datasets.mnist
print(type(mnist))
# 加载数据集
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# 查看几个数据的类型
print(type(x_train), type(y_train), type(x_test), type(y_test))
# 查看数据的shape
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 对数据做归一化，更容易训练
x_train, x_test = x_train / 255.0, x_test / 255.0
# 搭建模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  #目标变量为系数交叉熵分类损失函数：
              metrics=['accuracy'])
print(model.summary())
# fit函数就是执行训练的意思
# 注意这里直接把numpy的数组可以放进来训练，适合小数据集（大数据集用Datasets API)
model.fit(x_train, y_train, epochs=5)
# 评估结果
model.evaluate(x_test, y_test)
model.save(r'D:/PycharmProgram/total_model/my_model3.h5')

