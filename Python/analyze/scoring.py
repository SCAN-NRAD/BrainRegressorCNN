import csv
import typing
import io
import re
import numpy as np
from sklearn import metrics


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def parse_results_csv(csv_file) -> typing.Tuple[list, list, np.ndarray, np.ndarray]:
    params = None
    subjects = []
    predictions = []
    gt = []
    # name, col1, col1.gt, col2, col2.gt....
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if params is None:
                params = row[1::2]
            else:
                subjects.append(row[0])
                predictions.append(row[1::2])
                gt.append(row[2::2])

    predictions = np.array(predictions).astype(np.float32)
    gt = np.array(gt).astype(np.float32)

    return subjects, params, predictions, gt


def print_summary(subjects: list, params: list, predictions: np.ndarray, gt: np.ndarray, all_subjects=False) -> str:
    buff = io.StringIO()
    # scores per subject
    mape_scores_subject = []
    for idx, _ in enumerate(subjects):
        mape_scores_subject.append(mean_absolute_percentage_error(gt[idx], predictions[idx]))

    mape_scores_subject = np.array(mape_scores_subject)

    # scores per metric
    r2_scores_metric = []
    rmse_scores_metric = []
    for idx, _ in enumerate(params):
        r2_scores_metric.append(metrics.r2_score(gt[:, idx], predictions[:, idx]))
        rmse_scores_metric.append(np.sqrt(metrics.mean_squared_error(gt[:, idx], predictions[:, idx])))

    r2_scores_metric = np.array(r2_scores_metric)
    rmse_scores_metric = np.array(rmse_scores_metric)

    # print scores per metric
    print('{:<40}{:<10}  {}'.format('Metric', 'R^2', 'RMSE'), file=buff)
    print('------------------------------------------------------', file=buff)
    for idx, _ in enumerate(params):
        print('{:<40}{:.8f}  {:.3f}'.format(params[idx], r2_scores_metric[idx], rmse_scores_metric[idx]), file=buff)

    flop_metrics = np.argsort(r2_scores_metric)
    print(file=buff)

    # print top 5 and flop 5 subjects
    top_subjects = np.argsort(mape_scores_subject)
    n_top_subjects_5 = min(5, max(1, int(np.floor(len(top_subjects) / 3))))
    if all_subjects:
        n_top_subjects = len(top_subjects)
    else:
        n_top_subjects = n_top_subjects_5

    print('{:<40}  {}'.format('Flop 5 (by MAPE)', 'MAPE'), file=buff)
    print('--------------------------------------------------------------------------', file=buff)
    for idx in range(len(top_subjects)-1, len(top_subjects)-n_top_subjects-1, -1):
        subj_idx = top_subjects[idx]
        print('{:<40}  {:.5f}'.format(
            subjects[subj_idx], mape_scores_subject[subj_idx]), file=buff)

    if not all_subjects:
        print(file=buff)
        print('{:<40}  {}'.format('Top 5 (by MAPE)', 'MAPE'), file=buff)
        print('--------------------------------------------------------------------------', file=buff)
        for idx in range(0, n_top_subjects):
            subj_idx = top_subjects[idx]
            print('{:<40}  {:.5f}'.format(
                subjects[subj_idx], mape_scores_subject[subj_idx]), file=buff)

    print(file=buff)
    print(file=buff)
    print('R^2 Total:                {:.8f}'.format(metrics.r2_score(gt, predictions)), file=buff)

    # for cortex also report cortical Thickness and Curvature separately
    pattern = {'^(?!lh|rh).*': 'Subcortex', 'lh.*|rh.*': 'Cortex', '.*ThickAvg': 'ThickAvg', '.*MeanCurv': 'MeanCurv'}
    for p in pattern.keys():
        # get indices of params containing p
        indices = [idx for idx, elem in enumerate(params) if re.match(p, elem)]
        if len(indices) > 0:
            gt_sub = gt[:, indices]
            predictions_sub = predictions[:, indices]
            print('R^2 {:<9}:            {:.8f}'.format(pattern[p], metrics.r2_score(gt_sub, predictions_sub)), file=buff)

    return buff.getvalue(), r2_scores_metric


def parse_stats_csv(csv_file) -> typing.Tuple[dict, list]:
    metric_names = None
    morphometrics = dict()

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if metric_names is None:
                metric_names = row
            else:
                morphometrics[row[0].replace('.', '-')] = np.array(row[1:10]).astype(np.float32)

    return morphometrics, metric_names
