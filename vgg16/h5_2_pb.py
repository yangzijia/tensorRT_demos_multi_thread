#*-coding:utf-8-*

"""
将keras的.h5的模型文件，转换成TensorFlow的pb文件
"""
# ==========================================================
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from kerascv.model_provider import get_model
from keras import backend as K

from keras.models import load_model
import tensorflow as tf
import os
from keras import backend


def build_model_vgg16_v2(is_freeze=False, classes=2, image_w_h_size=224):
    # 定义模型poly学习策略
    base_model = VGG16(weights='imagenet', include_top=False, classes=classes,
                       input_shape=(image_w_h_size, image_w_h_size, 3))
    if is_freeze:
        for layer in base_model.layers:  # 保留前 freeze_layers_number 层网络全部参数
            layer.trainable = False
    x = base_model.output  # 自定义网络
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)  # 全连接层，激活函数elu
    x = Dropout(0.5)(x)  # Droupout 0.6
    x = Dense(4096, activation='relu', name='fc2')(x)  # 全连接层，激活函数elu
    x = Dropout(0.5)(x)  # Droupout 0.6
    predictions = Dense(classes, activation='softmax', name='predictions')(x)  # 输出层，指定类数

    model = Model(inputs=base_model.input, outputs=predictions)  # 新网络=预训练网络+自定义网络
    return model


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    """.h5模型文件转换成pb模型文件
    Argument:
        h5_model: str
            .h5模型文件
        output_dir: str
            pb模型文件保存路径
        model_name: str
            pb模型文件名称
        out_prefix: str
            根据训练，需要修改
        log_tensorboard: bool
            是否生成日志文件
    Return:
        pb模型文件
    """
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = backend.get_session()

    from tensorflow.python.framework import graph_util, graph_io
    # 写入pb模型文件
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    # 输出日志文件
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)


if __name__ == '__main__':
    #  .h模型文件路径参数
    input_path = 'aaa'
    weight_file = "vgg16_weights.h5"
    weight_file_path = os.path.join(input_path, weight_file)
    output_graph_name = weight_file[:-3] + '.pb'

    #  pb模型文件输出输出路径
    output_dir = os.path.join(os.getcwd(), "")

    #  加载模型
    # h5_model = load_model(weight_file_path)
    h5_model = build_model_vgg16_v2()
    h5_model.load_weights(weight_file_path)
    h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
    print('Finished')