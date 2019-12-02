import scipy
import sys
import os
import numpy as np
import tensorflow as tf

batch_size = 1
if not tf.test.is_built_with_cuda():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# SpaceNet
# test_img=sorted(glob.glob(r'/media/lc/vge_lc/spacenet/shanghai_vegas_test_result/test_image/*.png'))
# Urban
# test_img = np.array(sorted(glob.glob(r'/home/lc/Jupyter_projects/resatt/Urban 3D Challenge Data/d_test/img/*.png')))

def predict(test_img_path,pb_path,save_path):
    name=os.listdir(test_img_path)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
            output_tensor_name = sess.graph.get_tensor_by_name("conv2d_149/Conv2D:0")

            for j in range(0, len(name)):
                x_batch = os.path.join(test_img_path,name[j])
                i = os.path.basename(x_batch)
                x_batch = scipy.misc.imread(x_batch) / 255.0
                x_batch = np.expand_dims(x_batch, axis=0)
                predict = sess.run(output_tensor_name, feed_dict={input_image_tensor: x_batch})
                predict[predict < 0.5] = 0
                predict[predict >= 0.5] = 1
                result = np.squeeze(predict)
                i = i.split('.')[0]
                scipy.misc.imsave(save_path+'/{}.png'.format(i), result)


# test_img_path=sys.argv[0]
# pb_path =sys.argv[1]
# save_path=sys.argv[2]

test_img_path="/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/test/image"
pb_path='/home/lc/Jupyter_projects/resatt/MAPNet/checkpoint_ori/mapnet_model_whu.pb'
save_path='./test_result_temp'
predict(test_img_path,pb_path,save_path)

