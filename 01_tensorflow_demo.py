import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensor_demo():
    """
    张量的演示：numpy中的ndarray类型
    :return:
    """
    con_a = tf.constant(4.0)
    con_b = tf.constant([3, 4, 5])

    print(con_a)
    print(con_b)

    # 占位符
    ph_a = tf.placeholder(tf.int32, shape=(None, 2))
    print('ph_a:\n', ph_a)

    # 修改形状（保持元素个数一致）
    # new_a = tf.reshape(ph_a, shape=(4, 2))
    ph_a.set_shape(shape=(5, 2))
    print('new_a:\n', ph_a)

    return None


def variable_demo():
    """
    变量的演示：更新参数、保存模型
    :return:
    """
    with tf.variable_scope('name', reuse=tf.AUTO_REUSE):
        a = tf.Variable(initial_value=2, name='var_a')
        b = tf.Variable(initial_value=4, name='var_b')
        add = tf.add(a, b)

    # 变量初始化
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)  # 通过初始化对象申请内存空间
        add = sess.run([a, b, add])
        print(add)

    return None


def graph_demo():
    """
    tensor流图
    :return:
    """

    # with tf.variable_scope('sum', reuse=tf.AUTO_REUSE):  # reuse参数表示开启共享
    with tf.variable_scope('sum'):
        con_a = tf.constant(10, name='con_a')
        con_b = tf.constant(20, name='con_b')
        con_sum = tf.add(con_a, con_b, name='con_sum')
        print('con_sum = ', con_sum)

    # 默认图
    default_g = tf.get_default_graph()
    print('获取默认图:\n', default_g)

    # 属性
    print('con_a的graph:\n', con_a.graph)
    print('con_b的graph:\n', con_b.graph)
    print('con_sum的graph:\n', con_sum.graph)

    # 开启会话
    with tf.Session() as sess:
        print('在sess中的con_sum:\n', sess.run(con_sum))
        print('会话属性:\n', sess.graph)

        # 写入文件
        tf.summary.FileWriter('./test_default_graph', graph=sess.graph)

    return None


def graph1_demo():
    """
    自定义图
    :return:
    """

    # 自定义图
    new_g = tf.Graph()
    print('new_g:\n', new_g)

    # 在自定义图中定义数据和操作
    with new_g.as_default():
        new_a = tf.constant(100, name='new_a')
        new_b = tf.constant(500, name='new_b')
        new_sum = tf.add(new_a, new_b, name='new_sum')

    print('查看new_a的属性:\n', new_a.graph)
    print('查看new_b的属性:\n', new_b.graph)
    print('查看new_sum的属性:\n', new_sum.graph)

    with tf.Session(graph=new_g) as sess:
        new_sum_value = sess.run(new_sum)
        print('new_sum_value:\n', new_sum_value)
        print('查看sess的图属性:\n', sess.graph)

        # 写入文件
        tf.summary.FileWriter('./self_graph', graph=sess.graph)

    return None


if __name__ == '__main__':
    # 张量
    # tensor_demo()

    # 变量
    # variable_demo()

    # 默认画图
    # graph_demo()

    # 自定义画图
    graph1_demo()
