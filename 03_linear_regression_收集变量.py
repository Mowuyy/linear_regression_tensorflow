import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def linear_regression():
    """
    自实现简单的线性回归：

        需求：
            y = 0.8x + 0.7

            100样本中，每一个样本只有一个特征，对应一个目标值，
            用随机生成张量的方式x，根据准备好的关系得出y

            X: shape=(100, 1)     y_true: shape=(100, 1)

            矩阵乘法可知形状 => X(100, 1) * (1, 1) = (100, 1)

    :return:
    """

    # 1、构造数据
    with tf.variable_scope('prepare_data'):
        X = tf.random_normal((100, 1), name="X")
        y_true = tf.matmul(X, [[0.8]]) + 0.7

    # 2、假定线性模型
    with tf.variable_scope('create_model'):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)), name='Weights')
        bias = tf.Variable(initial_value=tf.random_normal(shape=(1,)), name='Bias')
        # bias = tf.Variable(initial_value=tf.random_normal(shape=(1,)), name='Bias', trainable=False)  # trainable表示该参数是否参与训练
        y_predict = tf.matmul(X, weights) + bias

    # 3、确定损失函数：均方误差
    with tf.variable_scope('loss_function'):
        error = tf.reduce_mean(tf.square(y_predict - y_true))

    # 4、优化损失函数：梯度下降
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 变量初始化
    init = tf.global_variables_initializer()

    # 创建事件文件（自动创建目录）：由于保存的是图、没有具体数据，因此可不放在会话中
    file_writer = tf.summary.FileWriter('./temp/linear_regression', graph=optimizer.graph)

    # a、收集变量
    tf.summary.scalar("error", error)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", bias)

    # b、合并变量
    merged = tf.summary.merge_all()

    # 开启会话
    with tf.Session() as sess:
        # 变量初始化
        sess.run(init)

        print("训练前随机初始化的权重值为%f，偏置为%f\n" % (weights.eval(), bias.eval()))

        # 在会话中创建事件文件（自动床架目录）
        # file_writer = tf.summary.FileWriter('./temp/linear_regression', graph=sess.graph)

        # c、运行优化器
        for i in range(700):
            sess.run(optimizer)
            print("第%d次训练后权重值为%f, 偏置为%f" % (i+1, weights.eval(), bias.eval()))

            # d、运行合并变量
            summary = sess.run(merged)

            # e、把变量添加至事件文件
            file_writer.add_summary(summary, i)

    return None


if __name__ == '__main__':
    linear_regression()

"""
注意：
    学习率 和 优化器 要配合使用
    1、学习率太大 ==> 梯度爆炸
    2、学习率太小 ==> 梯度消失
    
    方案：
        1、重新设计网络 
        2、调整学习率
        3、使用梯度截断（在训练过程中检查和限制梯度的大小）
        4、使用激活 函数
"""
