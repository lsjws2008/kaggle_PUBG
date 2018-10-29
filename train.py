import tensorflow as tf
import numpy as np

from generator.generator import generator
from models.default_model import default
from tools.losses import mse_loss
from tools.tool import initialize_uninitialized, Tenordered_decay
from global_config.global_config import config

if __name__ == '__main__':
    cfg = config()
    train_generator = generator(cfg.batch,
                                cfg.train_csv)
    # test_generator = generator(cfg.batch,
    #                            cfg.test_csv)

    with tf.Session() as sess:
        train_P = tf.placeholder(tf.float32,
                                 [None, cfg.input_shape, 1])
        target_P = tf.placeholder(tf.float32,
                                  [None])
        learning_rate = tf.placeholder(tf.float32,
                                       [None])

        train_model = default(train_P)

        net_out = train_model.inputs
        loss = mse_loss(net_out,
                       target_P)

        train_step = tf.train.MomentumOptimizer(learning_rate[0],
                                                cfg.momentum). \
            minimize(loss)
        initialize_uninitialized(sess)
        model_af = tf.train.Saver()

        for epoch in range(cfg.epoch):

            train_data, train_target = next(train_generator)
            learning_rate_edit = cfg.learning_rate/Tenordered_decay(epoch)
            sess.run(train_step, feed_dict={train_P: train_data,
                                            target_P: train_target,
                                            learning_rate: [learning_rate_edit]})

            if epoch % 10 == 0:
                test_data, test_target = next(train_generator)
                [p_loss, out] = sess.run([loss, net_out],
                                         feed_dict={train_P: test_data,
                                                    target_P: test_target})

                print('Epoch: {0:d}, loss: {1:.3f}, mean output: {2:.3f}, mean target: {3:.3f}'.
                      format(epoch,
                             p_loss,
                             np.mean(out),
                             np.mean(train_target)), end="\r")

            if (epoch + 1) % 10000 == 0:
                model_af.save(sess,
                              'save_model//'+'save_net_'+str(epoch + 1)+'.ckpt')
