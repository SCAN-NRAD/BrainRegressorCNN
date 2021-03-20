import os
import argparse
import numpy as np
import tensorflow as tf

from train import training
import data.storage as data_storage
import miapy.data.extraction as miapy_extr
from config import Configuration
import analyze.scoring as analyze_score


def batch_to_feed_dict(x_placeholder, train_placeholder, image):
    return {x_placeholder: image[np.newaxis, :, :, :].astype(np.float32), train_placeholder: False}


def main_predict(cfg_file, checkpoint_file, hdf_file, subjects_file, out_csv):
    cfg = Configuration.load(cfg_file)

    trainer = training.Trainer(cfg, 0)
    data_store = data_storage.DataStore(hdf_file if hdf_file else cfg.hdf_file)

    if subjects_file:
        with open(subjects_file, 'r') as file:
            subjects = [s.rstrip() for s in file.readlines()]
    else:
        subjects = [s['subject'] for s in data_store.dataset]

    validation_loader = data_store.get_loader(cfg.batch_size_eval, subjects, 0)
    validation_extractor = miapy_extr.ComposeExtractor(
        [miapy_extr.DataExtractor(),
         miapy_extr.SelectiveDataExtractor(category=data_storage.STORE_MORPHOMETRICS),
         miapy_extr.SubjectExtractor(),
         data_storage.DemographicsExtractor()])

    data_store.dataset.set_extractor(validation_extractor)

    column_values, column_names = data_store.get_all_metrics()
    trainer._regression_column_ids = np.array([column_names.index(name) for name in cfg.regression_columns])
    trainer._regression_column_multipliers = np.array(cfg.z_column_multipliers)

    with tf.Graph().as_default() as graph:
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            print('Using checkpoint {}'.format(checkpoint_file))
            saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
            saver.restore(sess, checkpoint_file)

            net = graph.get_tensor_by_name('NET/model:0')
            x_placeholder = graph.get_tensor_by_name('x:0')
            y_placeholder = graph.get_tensor_by_name('y:0')
            d_placeholder = graph.get_tensor_by_name('d:0')
            is_train_placeholder = graph.get_tensor_by_name('is_train:0')
            epoch_checkpoint = graph.get_tensor_by_name('epoch:0')

            print('Epoch from checkpoint: {}'.format(epoch_checkpoint.eval()))

            predictions, gt, pred_subjects = trainer.predict(sess,
                                                             net,
                                                             validation_loader,
                                                             x_placeholder,
                                                             y_placeholder,
                                                             d_placeholder,
                                                             is_train_placeholder)

            trainer.write_results_csv(out_csv, predictions, gt, pred_subjects)
            if len(pred_subjects) > 1:
                s, _ = analyze_score.print_summary(subjects, cfg.regression_columns, predictions, gt)
                print(s)

            if len(pred_subjects) != len(subjects):
                print("WARN: Number of subjects in predictions ({}) != given ({})".format(len(pred_subjects),
                                                                                          len(subjects)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate predictions based on a trained model')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (required for column multipliers, optional for hdf-file).'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint (trained model to load).'
    )

    parser.add_argument(
        '--hdf_file',
        type=str,
        required=False,
        help='Path to datafile (Optional, taken from config if not specified).'
    )

    parser.add_argument(
        '--subjects_file',
        type=str,
        required=False,
        help='Path to file with list of subjects to predict (Optional, take all in hdf-file if not specified).'
    )

    parser.add_argument(
        'results_csv',
        type=str,
        help='Path to results.csv to write.'
    )

    args = parser.parse_args()

    results_csv = args.results_csv
    if results_csv and not results_csv.startswith('/'):
        results_csv = os.path.join(os.getcwd(), results_csv)

    main_predict(args.config, args.checkpoint, args.hdf_file, args.subjects_file, results_csv)
