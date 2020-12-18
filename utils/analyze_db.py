from sql_writer import SQLWriter
import argparse
import numpy
import os
import logging
import datetime

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def plot_3d_graph(X, Y, Z, file_name):
    fig, ax = plt.subplots()

    # Plot the surface.
    im = ax.pcolormesh(X, Y, Z)
    fig.colorbar(im, ax=ax)

    ax.set_xlabel('learning rate')
    ax.set_ylabel('weight decay')

    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.suptitle('f1 macro colormap')

    # plt.show()
    fig.savefig(file_name)
    plt.close()


def get_all_results(db_path):
    sql_writer = SQLWriter(db_path=db_path)
    all_tasks = sql_writer.read(show=False)

    # construct 2d dictionaries
    result_dict = {}  # result_dict[lr][wd] = [{f1_macro_1, model_dir_1}, {f1_macro_2, model_dir_2}, ...]

    for task in all_tasks:
        time_stamp, task_args, best_tr_f1_macro, best_tr_acc, best_tr_epoch, best_va_f1_macro, \
        best_va_acc, best_va_epoch, current_epoch, tr_time, va_time, terminated = task
        # push into the corresponding dictionary, if there's no such key

        task_args = eval(task_args)
        lr = task_args.lr
        wd = task_args.weight_decay
        f1_macro = best_va_f1_macro
        model_dir = os.path.join(task_args.save_dir, time_stamp)

        if lr not in result_dict:
            result_dict[lr] = dict()
        if wd not in result_dict[lr]:
            result_dict[lr][wd] = list()
        result_dict[lr][wd].append({'f1_macro': f1_macro, 'model_dir': model_dir})

    # next use the max performance in all runs for each grid cell
    for lr, wd_dict in result_dict.items():
        for wd, result_lst in wd_dict.items():
            result_dict[lr][wd] = max(result_lst, key=lambda x: x['f1_macro'])

    lr_values = sorted(list(result_dict.keys()))
    wd_values = sorted(list(result_dict[lr_values[0]].keys()))

    return result_dict, lr_values, wd_values


def prepare_plot_data(result_dict, lr_values, wd_values):
    result_X = []
    result_Y = []
    result_Z = []

    for lr in lr_values:
        result_X.append([])
        result_Y.append([])
        result_Z.append([])
        for wd in wd_values:
            result_X[-1].append(lr)
            result_Y[-1].append(wd)
            result_Z[-1].append(result_dict[lr][wd]['f1_macro'])

    result_X = numpy.array(result_X)
    result_Y = numpy.array(result_Y)
    result_Z = numpy.array(result_Z)
    return result_X, result_Y, result_Z


def find_best_hyperparameter(result_dict):
    best_lr = 0
    best_wd = 0
    best_f1_macro = 0
    best_model_dir = ''

    for lr, wd_dict in result_dict.items():
        for wd, model_dict in wd_dict.items():
            if model_dict['f1_macro'] > best_f1_macro:
                best_lr = lr
                best_wd = wd
                best_f1_macro = model_dict['f1_macro']
                best_model_dir = model_dict['model_dir']

    return best_lr, best_wd, best_f1_macro, best_model_dir


def analyze_db(db_path, logger):
    logger.info('*' * 45 + ' analysis ' + '*' * 45)
    result_dict, lr_values, wd_values = get_all_results(db_path)
    logger.info(f'lr_values: {lr_values}')
    logger.info(f'wd_values: {wd_values}')
    logger.info(f'The results are:')
    for lr in lr_values:
        for wd in wd_values:
            logger.info(f"lr: {lr}, wd: {wd}, f1 macro: {result_dict[lr][wd]['f1_macro']}, model dir: {result_dict[lr][wd]['model_dir']}")

    result_X, result_Y, result_Z = prepare_plot_data(result_dict, lr_values, wd_values)
    fig_save_path = db_path + '.result_dict.png'
    plot_3d_graph(result_X, result_Y, result_Z, fig_save_path)

    best_lr, best_wd, best_f1_macro, best_model_dir = find_best_hyperparameter(result_dict)

    logger.info(f'the best performing hyperparameter is lr: {best_lr}, wd: {best_wd}')
    logger.info(f'the best f1 macro is: {best_f1_macro}')
    logger.info(f'the best model dir is {best_model_dir}')
    logger.info(f'the hyperparameter search figure is at {fig_save_path}')

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.info(f'the hyperparameter search log is at {handler.baseFilename}')


def start_logger(db_path):
    # starting logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(db_path + '.log.txt')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, required=True)
    args = parser.parse_args()

    logger = start_logger(args.db_path)

    analyze_db(args.db_path, logger)
