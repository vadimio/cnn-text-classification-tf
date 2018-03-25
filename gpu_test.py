import tensorflow as tf
from tensorflow.python.client import device_lib

print('\nTensor Flow Supported Device Detection')
print('--------------------------------------\n')

devices = device_lib.list_local_devices()

print('')

for device in devices:
    print(f'{device.device_type}: {device.name}, RAM: {device.memory_limit}')

gpu = [x for x in devices if x.device_type == 'GPU']

print('')

if gpu:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        print (sess.run(c))

else:
    print('No GPUs were detected')

print('\n')
