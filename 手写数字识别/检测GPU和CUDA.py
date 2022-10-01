import tensorflow as tf
print(tf.__version__)

#检测是否有GPU
GPU_device = tf.test.gpu_device_name()
print(GPU_device)

#检测GPU是否可用
print(tf.test.is_gpu_available())




