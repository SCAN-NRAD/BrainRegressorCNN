import os
import timeit
import importlib
import inspect
import datetime
import time
import csv
import re
import typing
import logging
import numpy as np
import tensorflow as tf
import torch
import random
from sklearn import model_selection
from sklearn import metrics

import miapy.data.extraction as miapy_extr

import data.storage as data_storage
import analyze.visualizing as visualize
import analyze.scoring as analyze_score

from config import Configuration

import model.net as model_net


TRAIN_TEST_SPLIT = 0.8

logger = logging.getLogger()


class Trainer:
    def __init__(self, cfg: Configuration, num_workers: int):
        self._cfg = cfg
        self._regression_column_multipliers = []
        self._regression_column_ids = []
        self._checkpoint_dir = None
        self._data_store = None
        self._num_workers = num_workers
        self._subjects_train = None
        self._subjects_validate = None
        self._subjects_test = None
        self._checkpoint_last_save = time.time()
        self._checkpoint_idx = -1
        self._best_r2_score = 0

    @property
    def checkpoint_dir(self) -> str:
        if self._checkpoint_dir is None:
            tstamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            self._checkpoint_dir = self._cfg.checkpoint_dir.replace('%m', self._cfg.model).replace('%t', tstamp)

        return self._checkpoint_dir

    def assign_subjects(self):
        if self._cfg.subjects_train_val_test_file:
            with open(self._cfg.subjects_train_val_test_file, 'r') as file:
                train, validate, test = file.readline().rstrip().split(';')
                self._subjects_train = train.split(',')
                self._subjects_validate = validate.split(',')
                self._subjects_test = test.split(',')
        else:
            # split data in train/validation/test
            self._subjects_train, val_test = model_selection.train_test_split(
                [s['subject'] for s in self._data_store.dataset],
                train_size=TRAIN_TEST_SPLIT,
                shuffle=True)
            self._subjects_validate, self._subjects_test = model_selection.train_test_split(val_test, train_size=0.5,
                                                                                            shuffle=True)
        logger.info('train/val/test: %s;%s;%s',
                    ','.join(self._subjects_train), ','.join(self._subjects_validate), ','.join(self._subjects_test))

    @staticmethod
    def set_seed(epoch: int):
        seed = 42 + epoch
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        torch.manual_seed(seed)

    def write_status(self, filename, msg=None):
        with open(os.path.join(self.checkpoint_dir, filename), 'w') as file:
            if msg:
                file.write(msg)

    def write_settings(self):
        # write subjects.txt
        with open(os.path.join(self.checkpoint_dir, 'subjects.txt'), 'w') as file:
            file.write('{};{};{}'.format(
                ','.join(self._subjects_train), ','.join(self._subjects_validate), ','.join(self._subjects_test)))

        # write network.py
        net_name = '{}.py'.format(self._cfg.model)
        with open(os.path.join(self.checkpoint_dir, net_name), 'w') as file:
            file.write(inspect.getsource(self.get_python_obj(self._cfg.model)))

        # write config.json
        # make copy of config before making changes
        cfg_tmp = Configuration()
        cfg_tmp.from_dict(self._cfg.to_dict())
        cfg_tmp.checkpoint_dir = self.checkpoint_dir
        cfg_tmp.subjects_train_val_test_file = os.path.join(self.checkpoint_dir, 'subjects.txt')
        cfg_tmp.z_column_multipliers = self._regression_column_multipliers.tolist()
        cfg_tmp.save(os.path.join(self.checkpoint_dir, 'config.json'))

    def extract_columns_batch(self, batch):
        return np.stack(batch[data_storage.STORE_MORPHOMETRICS], axis=0)[:, self._regression_column_ids]

    def batch_to_feed_dict(self, x_placeholder, y_placeholder, d_placeholder, train_placeholder, batch, is_train) -> dict:
        age = np.concatenate(batch[data_storage.STORE_DEMOGRAPHIC_AGE])
        sex = np.concatenate(batch[data_storage.STORE_DEMOGRAPHIC_SEX])
        demographics = np.stack([age, sex], axis=1).astype(np.float32)

        feed_dict = {x_placeholder: np.stack(batch[data_storage.STORE_IMAGES], axis=0).astype(np.float32),
                     d_placeholder: demographics,
                     train_placeholder: is_train}
        if is_train:
            feed_dict[y_placeholder] = self.extract_columns_batch(batch) / self._regression_column_multipliers

        return feed_dict

    def get_python_obj(self, model_function):
        mod_name, func_name = model_function.rsplit('.', 1)
        mod = importlib.import_module(mod_name)

        return getattr(mod, func_name)

    def get_transform(self):
        transform = None
        if self._cfg.data_augment_transform:
            # e.g. 'data.preprocess.RandomRotateShiftTransform:1:15'
            params = self._cfg.data_augment_transform.split(':')
            do_rotate = True
            shift = 0
            if len(params) > 1:
                do_rotate = int(params[1]) > 0
                shift = int(params[2])

            transform = self.get_python_obj(params[0])(do_rotate, shift)

        return transform

    def evaluate_loss(self, sess, loss, data_loader, x, y, d, is_train):
        sum_cost = 0
        num_samples = 0
        for batch in data_loader:
            print('+', end='', flush=True)
            feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, True)
            sum_cost = sum_cost + sess.run(loss, feed_dict=feed_dict) * len(batch)
            num_samples += len(batch)

        print()
        return sum_cost / num_samples

    def predict(self, sess, net, data_loader, x, y, d, is_train):
        predictions = None
        gt_labels = None
        subject_names = []
        for batch in data_loader:
            print('*', end='', flush=True)
            feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, False)
            predicted_labels = sess.run(net, feed_dict=feed_dict)
            gt = self.extract_columns_batch(batch)
            subject_names.extend(batch['subject'])
            if predictions is None:
                predictions = predicted_labels
                gt_labels = gt
            else:
                predictions = np.concatenate((predictions, predicted_labels))
                gt_labels = np.concatenate((gt_labels, gt))

        print()
        if predictions is None:
            return [], [], []
        else:
            return predictions * self._regression_column_multipliers, gt_labels, subject_names

    def write_results_csv(self, file_name: str, y: np.ndarray, gt: np.ndarray, subjects: typing.List[str]):
        file_path = file_name if file_name.startswith('/') else os.path.join(self.checkpoint_dir, file_name)
        with open(file_path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            row = ['name']
            [row.extend([c, '{}.gt'.format(c)]) for c in self._cfg.regression_columns]
            writer.writerow(row)

            for idx, subject in enumerate(subjects):
                row = [subject]
                [row.extend([y[idx, col], gt[idx, col]]) for col in range(y.shape[1])]
                writer.writerow(row)

    def checkpoint_safer(self, sess, saver, epoch_checkpoint, epoch, best_r2_score_checkpoint, force_safe=False, r2_score=None):
        # persist epoch
        sess.run(epoch_checkpoint.assign(tf.constant(epoch)))

        # checkpoint of best r2 score
        if r2_score and r2_score > self._best_r2_score:
            # persist best r2 score
            sess.run(best_r2_score_checkpoint.assign(tf.constant(r2_score)))
            path = os.path.join(self.checkpoint_dir, 'checkpoint-best-r-2.ckpt')
            logger.info('Saving new best checkpoint %s', path)
            saver.save(sess, path)
            self._best_r2_score = r2_score

        # periodic save
        now = time.time()
        if self._checkpoint_last_save + self._cfg.checkpoint_save_interval < now or epoch >= self._cfg.epochs - 1 or force_safe:
            self._checkpoint_idx = (self._checkpoint_idx + 1) % self._cfg.checkpoint_keep
            path = os.path.join(self.checkpoint_dir, 'checkpoint-{}.ckpt'.format(self._checkpoint_idx))
            logger.info('Saving checkpoint %s', path)
            saver.save(sess, path)
            self._checkpoint_last_save = now

    def train(self):
        self.write_status('status_init')
        start_time = timeit.default_timer()
        self.set_seed(epoch=0)

        transform = self.get_transform()
        self._data_store = data_storage.DataStore(self._cfg.hdf_file, transform)
        dataset = self._data_store.dataset

        self.assign_subjects()

        # prepare loaders and extractors
        training_loader = self._data_store.get_loader(self._cfg.batch_size, self._subjects_train, self._num_workers)
        validation_loader = self._data_store.get_loader(self._cfg.batch_size_eval, self._subjects_validate, self._num_workers)
        testing_loader = self._data_store.get_loader(self._cfg.batch_size_eval, self._subjects_test, self._num_workers)

        train_extractor = miapy_extr.ComposeExtractor(
            [miapy_extr.DataExtractor(),
             miapy_extr.SelectiveDataExtractor(category=data_storage.STORE_MORPHOMETRICS),
             miapy_extr.SubjectExtractor(),
             data_storage.DemographicsExtractor()])

        dataset.set_extractor(train_extractor)

        # read all labels to calculate multiplier
        column_values, column_names = self._data_store.get_all_metrics()
        self._regression_column_ids = np.array([column_names.index(name) for name in self._cfg.regression_columns])
        self._regression_column_multipliers = np.max(np.abs(column_values[:, self._regression_column_ids]), axis=0)

        model_net.SCALE = float(self._data_store.get_intensity_scale_max())

        n_batches = int(np.ceil(len(self._subjects_train) / self._cfg.batch_size))

        logger.info('Net: {}, scale: {}'.format(inspect.getsource(self.get_python_obj(self._cfg.model)), model_net.SCALE))
        logger.info('Train: {}, Validation: {}, Test: {}'.format(
            len(self._subjects_train), len(self._subjects_validate), len(self._subjects_test)))
        logger.info('Label multiplier: {}'.format(self._regression_column_multipliers))
        logger.info('n_batches: {}'.format(n_batches))
        logger.info(self._cfg)
        logger.info('checkpoints dir: {}'.format(self.checkpoint_dir))

        sample = dataset.direct_extract(train_extractor, 0)  # extract a subject to obtain shape

        with tf.Graph().as_default() as graph:
            self.set_seed(epoch=0)  # set again as seed is per graph

            x = tf.placeholder(tf.float32, (None,) + sample[data_storage.STORE_IMAGES].shape[0:], name='x')
            y = tf.placeholder(tf.float32, (None, len(self._regression_column_ids)), name='y')
            d = tf.placeholder(tf.float32, (None, 2), name='d')     # age, sex
            is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

            global_step = tf.train.get_or_create_global_step()
            epoch_checkpoint = tf.Variable(0, name='epoch')
            best_r2_score_checkpoint = tf.Variable(0.0, name='best_r2_score', dtype=tf.float64)

            net = self.get_python_obj(self._cfg.model)({'x': x, 'y': y, 'd': d, 'is_train': is_train})
            optimizer = None
            loss = None

            if self._cfg.loss_function == 'mse':
                loss = tf.losses.mean_squared_error(labels=y, predictions=net)
            elif self._cfg.loss_function == 'absdiff':
                loss = tf.losses.absolute_difference(labels=y, predictions=net)

            if self._cfg.learning_rate_decay_rate is not None and self._cfg.learning_rate_decay_rate > 0:
                learning_rate = tf.train.exponential_decay(self._cfg.learning_rate, global_step,
                                                           self._cfg.learning_rate_decay_steps,
                                                           self._cfg.learning_rate_decay_rate)
            else:
                learning_rate = tf.Variable(self._cfg.learning_rate, name='lr')

            if self._cfg.optimizer == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif self._cfg.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self._cfg.learning_rate)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):   # required for batch_norm
                train_op = optimizer.minimize(loss=loss, global_step=global_step)

            sum_lr = tf.summary.scalar('learning_rate', learning_rate)

            # collect variables to add to summaries (histograms, kernel weight images)
            sum_histograms = []
            sum_kernels = []
            kernel_tensors = []
            kernel_tensor_names = []
            for v in tf.global_variables():
                m = re.match('NET/(layer\d+_\w+)/(kernel|bias):0', v.name)
                if m:
                    var = graph.get_tensor_by_name(v.name)
                    sum_histograms.append(tf.summary.histogram('{}/{}'.format(m.group(1), m.group(2)), var))
                    #sum_histograms.append(tf.summary.histogram('{}/{}'.format(m.group(1), m.group(2)), tf.Variable(tf.zeros([1,1,1,1]))))

                    if m.group(2) == 'kernel' and m.group(1).endswith('conv'):
                        kernel_tensor_names.append(v.name)
                        h, w = visualize.get_grid_size(var.get_shape().as_list())
                        img = tf.Variable(tf.zeros([1, h, w, 1]))
                        kernel_tensors.append(img)
                        sum_kernels.append(tf.summary.image(m.group(1), img))

            summary_writer = tf.summary.FileWriter(os.path.join(self.checkpoint_dir, 'tb_logs'), tf.get_default_graph())

            init = tf.global_variables_initializer()
            # Saver keeps only 5 per default - make sure best-r2-checkpoint remains!
            # TODO: maybe set max_to_keep=None?
            saver = tf.train.Saver(max_to_keep=self._cfg.checkpoint_keep + 2)

            with tf.Session() as sess:
                sess.run(init)

                checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                if checkpoint:
                    # existing checkpoint found, restoring...
                    if not self._cfg.subjects_train_val_test_file:
                        msg = 'Continue training, but no fixed subject assignments found. ' \
                              'Set subjects_train_val_test_file in config.'
                        logger.error(msg)
                        raise RuntimeError(msg)

                    logger.info('Restoring from ' + checkpoint)
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')
                    saver.restore(sess, checkpoint)
                    if saver._max_to_keep < self._cfg.checkpoint_keep + 2:
                        msg = 'ERROR: Restored saver._max_to_keep={}, but self._cfg.checkpoint_keep={}'.format(saver._max_to_keep, self._cfg.checkpoint_keep)
                        print(msg)
                        logger.error(msg)
                        exit(1)
                        
                    sess.run(epoch_checkpoint.assign_add(tf.constant(1)))
                    self._best_r2_score = best_r2_score_checkpoint.eval()
                    logger.info('Continue with epoch %i (best r2: %f)', epoch_checkpoint.eval(), self._best_r2_score)
                    self._checkpoint_idx = int(re.match('.*/checkpoint-(\d+).ckpt', checkpoint).group(1))

                    # load column multipliers from file if available
                    cfg_file = os.path.join(self.checkpoint_dir, 'config.json')
                    if os.path.exists(cfg_file):
                        cfg_tmp = Configuration.load(cfg_file)
                        if len(cfg_tmp.z_column_multipliers) > 0:
                            logger.info('Loading column multipliers from %s', cfg_file)
                            self._regression_column_multipliers = np.array(cfg_tmp.z_column_multipliers)
                            logger.info('Label multiplier: {}'.format(self._regression_column_multipliers))
                else:
                    # new training, write config to checkpoint dir
                    self.write_settings()

                summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=global_step.eval())

                for epoch in range(epoch_checkpoint.eval(), self._cfg.epochs + 1):
                    self.set_seed(epoch=epoch)

                    self.write_status('status_epoch', str(epoch))
                    epoch_start_time = timeit.default_timer()
                    loss_sum = 0
                    r2_validation = None

                    # training (enable data augmentation)
                    self._data_store.set_transforms_enabled(True)
                    for batch in training_loader:
                        print('.', end='', flush=True)

                        feed_dict = self.batch_to_feed_dict(x, y, d, is_train, batch, True)

                        # perform training step
                        _, loss_step = sess.run([train_op, loss], feed_dict=feed_dict)
                        loss_sum = loss_sum + loss_step * len(batch)

                    # disable transformations (data augmentation) for validation
                    self._data_store.set_transforms_enabled(False)

                    # loss on training data
                    cost_train = loss_sum / len(self._subjects_train)

                    if epoch % self._cfg.log_num_epoch == 0:
                        # loss on validation set
                        cost_validation = self.evaluate_loss(sess, loss, validation_loader, x, y, d, is_train)
                        cost_validation_str = '{:.16f}'.format(cost_validation)
                    else:
                        print()
                        cost_validation = None
                        cost_validation_str = '-'

                    logger.info('Epoch:{:4.0f}, Loss train: {:.16f}, Loss validation: {}, lr: {:.16f}, dt={:.1f}s'.format(epoch, cost_train, cost_validation_str, learning_rate.eval(), timeit.default_timer() - epoch_start_time))

                    # don't write loss for first epoch (usually very high) to avoid scaling issue in graph
                    if epoch > 0:
                        # write summary
                        summary = tf.Summary()
                        summary.value.add(tag='loss_train', simple_value=cost_train)
                        if cost_validation:
                            summary.value.add(tag='loss_validation', simple_value=cost_validation)
                        summary_writer.add_summary(summary, epoch)

                        summary_op = tf.summary.merge([sum_lr])
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, epoch)

                    if epoch % self._cfg.log_eval_num_epoch == 0 and epoch > 0:
                        # calculate and log R2 score on training and validation set
                        eval_start_time = timeit.default_timer()
                        predictions_train, gt_train, subjects_train = self.predict(sess, net, training_loader, x, y, d, is_train)
                        predictions_validation, gt_validation, subjects_validation = self.predict(sess, net, validation_loader, x, y, d, is_train)
                        r2_train = metrics.r2_score(gt_train, predictions_train)
                        r2_validation = metrics.r2_score(gt_validation, predictions_validation)
                        logger.info('Epoch:{:4.0f}, R2 train: {:.3f}, R2 validation: {:.8f}, dt={:.1f}s'.format(epoch, r2_train, r2_validation, timeit.default_timer() - eval_start_time))

                        # write csv with intermediate results
                        self.write_results_csv('results_train-{0:04.0f}.csv'.format(epoch), predictions_train, gt_train, subjects_train)
                        self.write_results_csv('results_validate-{0:04.0f}.csv'.format(epoch), predictions_validation, gt_validation, subjects_validation)

                        summary = tf.Summary()
                        # average r2
                        summary.value.add(tag='r2_train', simple_value=r2_train)
                        summary.value.add(tag='r2_validation', simple_value=r2_validation)

                        # add r2 per metric
                        for idx, col_name in enumerate(self._cfg.regression_columns):
                            summary.value.add(tag='train/r2_{}'.format(col_name), simple_value=metrics.r2_score(gt_train[:, idx], predictions_train[:, idx], multioutput='raw_values'))
                            summary.value.add(tag='validation/r2_{}'.format(col_name), simple_value=metrics.r2_score(gt_validation[:, idx], predictions_validation[:, idx], multioutput='raw_values'))

                        summary_writer.add_summary(summary, epoch)

                    if epoch % self._cfg.visualize_layer_num_epoch == 0 and len(sum_histograms) > 0:
                        # write histogram summaries and kernel visualization
                        summary_op = tf.summary.merge(sum_histograms)
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, epoch)

                        if len(kernel_tensor_names) > 0:
                            for idx, kernel_name in enumerate(kernel_tensor_names):
                                # visualize weights of kernel layer from a middle slice
                                kernel_weights = graph.get_tensor_by_name(kernel_name).eval()
                                # make last axis the first
                                kernel_weights = np.moveaxis(kernel_weights, -1, 0)

                                if len(kernel_weights.shape) > 4:
                                    # 3d convolution, remove last (single) channel
                                    kernel_weights = kernel_weights[:, :, :, :, 0]

                                if kernel_weights.shape[3] > 1:
                                    # multiple channels, take example from middle slide
                                    slice_num = int(kernel_weights.shape[3] / 2)
                                    kernel_weights = kernel_weights[:, :, :, slice_num:slice_num + 1]

                                grid = visualize.make_grid(kernel_weights)[np.newaxis, :, :, np.newaxis]
                                sess.run(kernel_tensors[idx].assign(grid))

                            summary_op = tf.summary.merge(sum_kernels)
                            summary_str = sess.run(summary_op)
                            summary_writer.add_summary(summary_str, epoch)

                    summary_writer.flush()

                    if self._cfg.max_timelimit > 0 and (timeit.default_timer() - start_time > self._cfg.max_timelimit):
                        logger.info('Timelimit {}s exceeded. Stopping training...'.format(self._cfg.max_timelimit))
                        self.checkpoint_safer(sess, saver, epoch_checkpoint, epoch, best_r2_score_checkpoint,
                                              True, r2_validation)
                        self.write_status('status_timeout')
                        break
                    else:
                        # epoch done
                        self.checkpoint_safer(sess, saver, epoch_checkpoint, epoch, best_r2_score_checkpoint, False, r2_validation)

                summary_writer.close()
                logger.info('Training done.')
                self.write_status('status_done')

                if self._best_r2_score > 0:
                    # restore checkpoint of best R2 score
                    checkpoint = os.path.join(self.checkpoint_dir, 'checkpoint-best-r-2.ckpt')
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')
                    saver.restore(sess, checkpoint)
                    logger.info('RESTORED best-r-2 checkpoint. Epoch: {}, R2: {:.8f}'.format(
                                epoch_checkpoint.eval(),
                                best_r2_score_checkpoint.eval()))

                # disable transformations (data augmentation) for test
                self._data_store.set_transforms_enabled(False)

                predictions_train, gt_train, subjects_train = self.predict(sess, net, training_loader, x, y, d, is_train)
                predictions_test, gt_test, subjects_test = self.predict(sess, net, testing_loader, x, y, d, is_train)

                self.write_results_csv('results_train.csv', predictions_train, gt_train, subjects_train)
                self.write_results_csv('results_test.csv', predictions_test, gt_test, subjects_test)

                # Note: use scaled metrics for MSE and unscaled (original) for R^2
                if len(gt_train) > 0:
                    accuracy_train = metrics.mean_squared_error(
                        gt_train / self._regression_column_multipliers,
                        predictions_train / self._regression_column_multipliers)
                    r2_train = metrics.r2_score(gt_train, predictions_train)
                else:
                    accuracy_train = 0
                    r2_train = 0

                if len(subjects_test) > 0:
                    accuracy_test = metrics.mean_squared_error(
                        gt_test / self._regression_column_multipliers,
                        predictions_test / self._regression_column_multipliers)
                    r2_test = metrics.r2_score(gt_test, predictions_test)

                    s, _ = analyze_score.print_summary(subjects_test, self._cfg.regression_columns, predictions_test, gt_test)
                    logger.info('Summary:\n%s-------', s)

                    logger.info('TRAIN accuracy(mse): {:.8f}, r2: {:.8f}'.format(accuracy_train, r2_train))
                    logger.info('TEST  accuracy(mse): {:.8f}, r2: {:.8f}'.format(accuracy_test, r2_test))

                visualize.make_kernel_gif(self.checkpoint_dir, kernel_tensor_names)

