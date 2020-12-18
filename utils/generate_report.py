import argparse
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save_dir', type=str, required=True,
                    help='''Usually the timestamp folder location, such as '/data/deep-sentence-classifiers-data/2019_06_07_13_54_27_060955'.''')
parser.add_argument('--label', type=str, default='hard')
parser.add_argument('--mode', type=str, default='full')  # full or no_other
parser.add_argument('--conf', action='store_true')
parser.add_argument('--per_class', action='store_true')

args = parser.parse_args()

# starting logger
filename = os.path.join(args.save_dir, 'classification_report.txt')
if os.path.isfile(filename):
    os.remove(filename)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler
handler = logging.FileHandler(filename)
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def log_metrics(y_true, y_pred):
    """
    Print classification report, accuracy, and confusion matrix
    args:
        y_true: True labels
        y_pred: Predictions
    """

    logger.info(args)
    logger.info(classification_report(y_true, y_pred))

    function_lib = {'f1_score': lambda x, y: f1_score(x, y, average='macro'),
                    'precision_score': lambda x, y: precision_score(x, y, average='macro'),
                    'recall_score': lambda x, y: recall_score(x, y, average='macro'),
                    'accuracy_score': accuracy_score}
    for function_name, score_function in function_lib.items():
        score = score_function(y_true, y_pred)
        logger.info(f'{function_name}: {score}')
        if args.conf:
            conf_interval_ = get_conf_interval(
                y_true, y_pred, metric_f=score_function)
            logger.info(f'conf_interval_: {conf_interval_}.')
            if args.per_class:
                conf_per_class_ = conf_per_class(
                    y_true, y_pred, metric_f=score_function)
                logger.info(f'conf_per_class_: {conf_per_class_}.')
        logger.info('-' * 50)


def get_conf_interval(y_true, y_pred, metric_f, num_samps=1000, alpha=.05):
    """
    Purpose: calculate confidence interval for metric under the metric calculated by metric_f

    y_true::list : ground truth labels
    y_pred::list : predicted labels
    metric_f: function that produces  float, and y_true,y_pred as inputs
    num_samps::int : number of times to take a bootstrapped sample
    alpha:: float : in (0,1) probability that actual value is outside of lower and upper bound
    returns::(float,float):  lower and upper bounds for confidence interval
    """
    samp_idx = np.random.choice(range(len(y_true)), size=(num_samps, len(y_true)), replace=True)

    scores = []

    # get numsamp bootstrapped samples
    for j in range(num_samps):
        samp_y = [y_true[i] for i in samp_idx[j]]
        samp_y_pred = [y_pred[i] for i in samp_idx[j]]
        scores.append(metric_f(samp_y, samp_y_pred))

    lower_bound = int(num_samps * alpha)
    upper_bound = int(num_samps * (1 - alpha))

    scores.sort()
    return scores[lower_bound], scores[upper_bound]


def conf_per_class(y_true, y_pred, metric_f, num_samps=1000, alpha=.05):
    """
    Purpose: calculate confidence interval for metric under the metric calculated by metric_f

    y_true::list : ground truth labels
    y_pred::list : predicted labels
    metric_f: function that produces  float, and y_true,y_pred as inputs
    num_samps::int : number of times to take a bootstrapped sample
    alpha:: float : in (0,1) probability that actual value is outside of lower and upper bound
    returns::dict : key is class labels present in y_true, lower and upper bounds for confidence interval
    """

    results = dict()

    for c in set(y_true):
        y_true_class = [1 if x == c else 0 for x in y_true]
        y_pred_class = [1 if x == c else 0 for x in y_pred]
        results[c] = get_conf_interval(y_true_class, y_pred_class, metric_f, num_samps=num_samps, alpha=alpha)

    return results


if __name__ == '__main__':
    true = torch.load(os.path.join(args.save_dir, 'validation_labels.true.pt'))
    pred = torch.load(os.path.join(args.save_dir, 'validation_labels.pred.pt'))
    if args.label == 'hard':
        cls2ind = torch.load(os.path.join(args.save_dir, 'validation_labels.pred.pt'))

    log_metrics(true, pred)
