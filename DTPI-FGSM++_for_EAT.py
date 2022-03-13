"""Implementation of sample attack."""
# coding: utf-8
#/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import *
from numpy import pi, exp, sqrt
from attack_method import *
from tqdm import tqdm
from tensorpack import TowerContext
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, resnet_v1
import fdnets
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import os
import cv2
from PIL import ImageFilter

slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', './models', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_csv', 'dataset/dev_dataset.csv', 'Input directory with images.')

tf.flags.DEFINE_string('input_dir', 'dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', 'output/', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float('num_classes', 1001, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 20, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 5, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_float('amplification_factor', 10.0, 'To amplifythe step size.')

tf.flags.DEFINE_float('project_factor', 1.0, 'To control the weight of project term.')

tf.flags.DEFINE_float('temperature', 1.5, 'To soften the output probability distribution.')
FLAGS = tf.flags.FLAGS

num_of_K = 1.5625  # 100 / 64
T_kern = gkern(15, 3)
P_kern, kern_size = project_kern(3)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2.ckpt'),
    'resnet_v2_101': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt'),
    'vgg_16': os.path.join(FLAGS.checkpoint_path,'vgg_16.ckpt'),
    'resnet_v2_152': os.path.join(FLAGS.checkpoint_path,'resnet_v2_152.ckpt'),
    'adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'adv_inception_resnet_v2.ckpt'),
    'resnet_v2_50': os.path.join(FLAGS.checkpoint_path,'resnet_v2_50.ckpt')}


def graph(adv, y, t_y, i, x_max, x_min, grad, amplification):
    target_one_hot = tf.one_hot(t_y, 1001)
    true_one_hot = tf.one_hot(y, 1001)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    alpha_beta = alpha * FLAGS.amplification_factor
    gamma = alpha_beta * FLAGS.project_factor
    momentum = FLAGS.momentum
    num_classes = 1001

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, reuse = True)
    auxlogits_v3 = end_points_v3['AuxLogits']
    #
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, reuse = True)
    auxlogits_v4 = end_points_v4['AuxLogits']

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_152, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, reuse = True)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_101, end_points_resnet_101 = resnet_v2.resnet_v2_101(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, reuse = True)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_50, end_points_resnet_50 = resnet_v2.resnet_v2_50(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, reuse = True)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_Incres, end_points_IR = inception_resnet_v2.inception_resnet_v2(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, reuse = True)
    auxlogits_Incres = end_points_IR['AuxLogits']

    # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #     logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
    #         input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3', reuse = True)
    # auxlogits_ens3 = end_points_ens3_adv_v3['AuxLogits']
    #
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3', reuse = True)
    auxlogits_ens4 = end_points_ens4_adv_v3['AuxLogits']

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(FLAGS, adv), num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2', reuse = True)
    auxlogits_ensadv = end_points_ensadv_res_v2['AuxLogits']

    logits = (logits_resnet_152 + logits_v4 + logits_Incres + logits_v3 + logits_resnet_50 +\
              logits_resnet_101 + logits_ensadv_res_v2 + logits_ens4_adv_v3) / 8.0 / FLAGS.temperature
    auxlogits = (auxlogits_v4 + auxlogits_Incres + auxlogits_v3 + auxlogits_ensadv + auxlogits_ens4) / 5.0 / FLAGS.temperature

    target_cross_entropy = tf.losses.softmax_cross_entropy(target_one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)

    target_cross_entropy += 0.7 * tf.losses.softmax_cross_entropy(target_one_hot,
                                                    auxlogits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)

    noise = tf.gradients(target_cross_entropy, adv)[0]
    noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')
    # noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    # noise = momentum * grad + noise
    # Project cut noise
    amplification += alpha_beta * n_staircase_sign(noise, num_of_K)
    cut_noise = tf.clip_by_value(abs(amplification) - eps, 0.0, 10000.0) * tf.sign(amplification)
    projection = gamma * n_staircase_sign(project_noise(cut_noise, P_kern, kern_size), num_of_K)

    adv = adv - alpha_beta * n_staircase_sign(noise, num_of_K) - projection
    adv = tf.clip_by_value(adv, x_min, x_max)
    i = tf.add(i, 1)
    return adv, y, t_y, i, x_max, x_min, noise, amplification

def stop(adv, y, t_y, i, x_max, x_min, grad, amplification):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    mean_pert = 0.0
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    sum_fd1, sum_fd2, sum_fd3, sum_adv_v3, \
        sum_ens3_adv_v3, sum_ens4_adv_v3, sum_ensadv_res_v2 = 0, 0, 0, 0, 0, 0, 0
    sum_v3, sum_v4, sum_res152, sum_res101, sum_res50, sum_Incres, sum_ensemble = 0,0,0,0,0,0,0
    tf.logging.set_verbosity(tf.logging.INFO)


    with tf.Graph().as_default():
        # Prepare graph
        adv_img = tf.placeholder(tf.float32, shape = batch_shape)
        y = tf.placeholder(tf.int32, shape = batch_shape[0])
        t_y = tf.placeholder(tf.int32, shape = batch_shape[0])
        x_max = tf.clip_by_value(adv_img + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(adv_img - eps, -1.0, 1.0)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False)
        pre_v3 = tf.argmax(logits_v3, 1)

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_v4, end_points_v4 = inception_v4.inception_v4(
                adv_img, num_classes=num_classes, is_training=False)
        pre_v4 = tf.argmax(logits_v4, 1)

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet_152, end_points_resnet = resnet_v2.resnet_v2_152(
                adv_img, num_classes=num_classes, is_training=False)
        pre_resnet_152 = tf.argmax(logits_resnet_152, 1)

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet_101, end_points_resnet_101 = resnet_v2.resnet_v2_101(
                adv_img, num_classes=num_classes, is_training=False)
        pre_resnet_101 = tf.argmax(logits_resnet_101, 1)

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet_50, end_points_resnet_50 = resnet_v2.resnet_v2_50(
                adv_img, num_classes=num_classes, is_training=False)
        pre_resnet_50 = tf.argmax(logits_resnet_50, 1)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_Incres, end_points_IR = inception_resnet_v2.inception_resnet_v2(
                adv_img, num_classes=num_classes, is_training=False)
        pre_Inc_res = tf.argmax(logits_Incres, 1)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
        pre_ens3_adv_v3 = tf.argmax(logits_ens3_adv_v3, 1)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
        pre_ens4_adv_v3 = tf.argmax(logits_ens4_adv_v3, 1)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                adv_img, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
        pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)


        pre_ensemble_logit = tf.argmax((logits_v4 + logits_Incres + logits_resnet_152 + logits_v3 + logits_resnet_50 + logits_resnet_101
                                        + logits_ensadv_res_v2 + logits_ens4_adv_v3), 1)


        sum_v3, sum_v4, sum_res152, sum_res101, sum_res50, sum_Incres, sum_ensemble, sum_ens3_adv_v3, sum_ens4_adv_v3, sum_ensadv_res_v2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        amplification = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _, _ = tf.while_loop(stop, graph, [adv_img, y, t_y, i, x_max, x_min, grad, amplification])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_50'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['adv_inception_v3'])
            s3.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s4.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s5.restore(sess, model_checkpoint_map['inception_v4'])
            s6.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            s8.restore(sess, model_checkpoint_map['resnet_v2_152'])
            s9.restore(sess, model_checkpoint_map['resnet_v2_101'])
            s10.restore(sess, model_checkpoint_map['resnet_v2_50'])

            import pandas as pd
            dev = pd.read_csv(FLAGS.input_csv)

            for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
                images, filenames, True_label, Target_label = load_images(FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
                my_adv_images = sess.run(x_adv, feed_dict={adv_img: images, y: True_label, t_y: Target_label}).astype(np.float32)
                pre_v3_, pre_v4_, pre_resnet152_, pre_resnet101_, pre_resnet50_, pre_Inc_res_, \
                pre_ens3_adv_v3_, pre_ens4_adv_v3_, pre_ensadv_res_v2_, pre_ensemble_ \
                     = sess.run([pre_v3, pre_v4, pre_resnet_152, pre_resnet_101, pre_resnet_50, pre_Inc_res,
                                 pre_ens3_adv_v3, pre_ens4_adv_v3, pre_ensadv_res_v2, pre_ensemble_logit], feed_dict = {adv_img: my_adv_images})

                sum_v3 += (pre_v3_ == Target_label).sum()
                sum_v4 += (pre_v4_ == Target_label).sum()
                sum_res152 += (pre_resnet152_ == Target_label).sum()
                sum_res101 += (pre_resnet101_ == Target_label).sum()
                sum_res50 += (pre_resnet50_ == Target_label).sum()
                sum_Incres += (pre_Inc_res_ == Target_label).sum()
                sum_ens3_adv_v3 += (pre_ens3_adv_v3_ == Target_label).sum()
                sum_ens4_adv_v3 += (pre_ens4_adv_v3_ == Target_label).sum()
                sum_ensadv_res_v2 += (pre_ensadv_res_v2_ == Target_label).sum()
                sum_ensemble += (pre_ensemble_ == Target_label).sum()
                save_images(my_adv_images, filenames, FLAGS.output_dir)

            print('sum_v3 = {}'.format(sum_v3))
            print('sum_v4 = {}'.format(sum_v4))
            print('sum_res2 = {}'.format(sum_res152))
            print('sum_res1 = {}'.format(sum_res101))
            print('sum_res1 = {}'.format(sum_res50))
            print('sum_Incres_v2 = {}'.format(sum_Incres))
            print('sum_ens3_adv_v3 = {}'.format(sum_ens3_adv_v3))
            print('sum_ens4_adv_v3 = {}'.format(sum_ens4_adv_v3))
            print('sum_ensadv_Incres_v2 = {}'.format(sum_ensadv_res_v2))
            print('sum_ensmeble = {}'.format(sum_ensemble))




if __name__ == '__main__':
    tf.app.run()
