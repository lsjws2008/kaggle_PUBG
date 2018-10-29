import tensorflow as tf
import numpy as np

from generator.generator import generator
from models.default_model import default
from tools.losses import mse_loss
from tools.tool import initialize_uninitialized
from global_config.global_config import config

if __name__ == '__main__':
    cfg = config()
    test_generator = generator(cfg.batch,
                               cfg.test_csv)

    with tf.Session() as sess:
        train_P = tf.placeholder(tf.float32,
                                 [None, cfg.input_shape])

        train_model = default(train_P)

        net_out = train_model.inputs

        model_pr = tf.train.Saver()
        model_pr.restore(sess, 'save_model\\100000save_net.ckpt')

        with open('datas//upload.csv', 'w') as f:
            f.write('Id,winPlacePerc\n')

        for epoch in range(cfg.epoch):

            test_data, IDs = next(test_generator)
            output = sess.run(net_out, feed_dict={train_P:test_data})
            print(IDs[0])
            with open('datas//upload.csv', 'a') as f:

                for i, j in zip(IDs, output):
                    f.write(str(i)+','+str(j[0])+'\n')
