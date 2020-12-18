import pandas
import argparse
import os
import subprocess
import logging
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import torch
import math
from pdb import set_trace
from collections import OrderedDict, defaultdict
from multiprocessing import Pool, current_process
from sklearn.model_selection import ParameterGrid
import operator
import pickle
import numpy
from copy import deepcopy
from sklearn.preprocessing import MultiLabelBinarizer

################### mutable code starts here
STARTER_PACKS_DATA = ['CenturyLink', 'Hawaiian', 'Telstra']
#os.environ['TORCH_MODEL_ZOO'] = '/home/jovyan/embeddings/deep-sentence-classifiers/tmp/'


def decode_parameter_tuple(parameter_tuple):
    parameter_dict = {'encoder': parameter_tuple[0], 'classifier': parameter_tuple[1], 'embedding_type': parameter_tuple[2], 'data': parameter_tuple[3], 'language': parameter_tuple[4]}
    parameter_dict['tr'] = []
    parameter_dict['va'] = []
    parameter_dict['te'] = []
    if 'xlt' in parameter_dict['language']:
        parameter_dict['LABEL_COLUMN_NAME'] = 'answer'
        if parameter_dict['language'] == 'xlt-eng-esp':
            parameter_dict['tr'].append('usic_data/tr.json')
            parameter_dict['va'].append('usic_data/va.json')
            parameter_dict['te'].append('usic_spanish_data/te.json')
        if parameter_dict['language'] == 'xlt-eng-por':
            parameter_dict['tr'].append('usic_data/tr.json')
            parameter_dict['va'].append('usic_data/va.json')
            parameter_dict['te'].append('usic_portuguese_data/te.json')
    elif parameter_dict['data'] == 'usic':
        parameter_dict['LABEL_COLUMN_NAME'] = 'answer'
        if 'eng' in parameter_dict['language']:
            parameter_dict['tr'].append('usic_data/tr.json')
            parameter_dict['va'].append('usic_data/va.json')
        if 'esp' in parameter_dict['language']:
            parameter_dict['tr'].append('usic_spanish_data/tr.json')
            parameter_dict['va'].append('usic_spanish_data/va.json')
        if 'por' in parameter_dict['language']:
            parameter_dict['tr'].append('usic_portuguese_data/tr.json')
            parameter_dict['va'].append('usic_portuguese_data/va.json')
        if parameter_dict['language'].startswith('eng'):
            parameter_dict['te'].append('usic_data/te.json')
        if parameter_dict['language'].startswith('esp'):
            parameter_dict['te'].append('usic_spanish_data/te.json')
        if parameter_dict['language'].startswith('por'):
            parameter_dict['te'].append('usic_portuguese_data/te.json')
    elif parameter_dict['data'] in STARTER_PACKS_DATA:
        parameter_dict['tr'].append(f'''preprocessed_data/{parameter_dict['data']}/tr.json''')
        parameter_dict['va'].append(f'''preprocessed_data/{parameter_dict['data']}/va.json''')
        parameter_dict['te'].append(f'''preprocessed_data/{parameter_dict['data']}/te_split.json''')
        parameter_dict['va_threshold'] = f'''preprocessed_data/{parameter_dict['data']}/va_threshold.json'''
        parameter_dict['LABEL_COLUMN_NAME'] = 'intent'
    elif parameter_dict['data'] == 'stackoverflow':
        parameter_dict['LABEL_COLUMN_NAME'] = 'tags'
        if 'eng' in parameter_dict['language']:
            parameter_dict['tr'].append('stackoverflow_data/eng/posts_train.csv')
            parameter_dict['va'].append('stackoverflow_data/eng/posts_val.csv')
        if parameter_dict['language'].startswith('eng'):
            parameter_dict['te'].append('stackoverflow_data/eng/posts_test.csv')
    else:
        raise NotImplementedError
    return parameter_dict
################### mutable code ends here


def get_model_path(parameter_tuple):
    return os.path.join(args.out_path, '_'.join([str(x) for x in parameter_tuple]))


def read_csv_json(file_name: str):
    if file_name.endswith('json') or file_name.endswith('jsonl'):
        df = pandas.read_json(file_name, lines=True)
    elif file_name.endswith('csv'):
        df = pandas.read_csv(file_name)
    else:
        raise NotImplementedError(f'failed when trying to read {file_name}')
    return df


def parallel_train_eval(parameter_tuple):
    model_path = get_model_path(parameter_tuple)
    if not os.path.isdir(args.out_path):
        os.mkdir(model_path)
    gpu_id = current_process()._identity[0] - 1

    print(gpu_id)
    parameter_dict = decode_parameter_tuple(parameter_tuple)
    if parameter_dict['encoder'].startswith('elmo'):
        parameter_dict['encoder_base_name'] = 'elmo'
        if parameter_dict['encoder'] == 'elmo-small':
            parameter_dict['elmo_option_file'] = 'elmo-small/config.json'
            parameter_dict['elmo_weights_file'] = 'elmo-small/elmo.hdf5'
        elif parameter_dict['encoder'] == 'elmo-original':
            parameter_dict['elmo_option_file'] = 'elmo-original/config.json'
            parameter_dict['elmo_weights_file'] = 'elmo-original/elmo.hdf5'
        elif parameter_dict['encoder'] == 'elmo-small-lp' or parameter_dict['encoder'] == 'elmo':  # parameter_dict['encoder'] == 'elmo' for backward compatibility
            parameter_dict['elmo_option_file'] = '11_brand_elmo_remove_html/config.json'
            parameter_dict['elmo_weights_file'] = '11_brand_elmo_remove_html/elmo.hdf5'
        elif parameter_dict['encoder'] == 'elmo-smaller-lp':
            parameter_dict['elmo_option_file'] = '11_brand_elmo_even_smaller/config.json'
            parameter_dict['elmo_weights_file'] = '11_brand_elmo_even_smaller/elmo.hdf5'
    else:
        parameter_dict['encoder_base_name'] = parameter_dict['encoder']

    # start training, output to a folder
    logger.info(f'training {model_path}')
    print(parameter_dict['tr'])
    print(' '.join(parameter_dict['tr']))
    cmd = ['python', 'train.py',
           '--gpu_id', str(gpu_id),
           '--tr']
    cmd += parameter_dict['tr']
    cmd += ['--va']
    cmd += parameter_dict['va']
    cmd += ['--save_dir', os.path.abspath(model_path),
           '--encoder', parameter_dict['encoder_base_name'],
           '--classifier', parameter_dict['classifier'],
           '--embedding_type', parameter_dict['embedding_type'],
           '--label_column_name', parameter_dict['LABEL_COLUMN_NAME']]
    if parameter_dict['encoder'].startswith('elmo'):
        cmd += ['--elmo_option_file', parameter_dict['elmo_option_file'],
                '--elmo_weights_file', parameter_dict['elmo_weights_file']]
    logger.info('(' * 50 + ')' * 50)
    logger.info(' '.join(cmd))
    logger.info('(' * 50 + ')' * 50)
    print('CALL TO TRAIN')
    print(' '.join(cmd))
    subprocess.run(cmd)
    
    # validation set evaluation
    true = torch.load(os.path.join(model_path, 'validation_labels.true.pt'))
    pred = torch.load(os.path.join(model_path, 'validation_labels.pred.pt'))
    logger.info('best model performance on va set')
    logger.info(classification_report(y_true=true, y_pred=pred))
    f1_macro_this_fold = f1_score(y_true=true, y_pred=pred, average='macro')
    logger.info(f'f1 this fold = {f1_macro_this_fold}')

    return parameter_tuple, f1_macro_this_fold


def parallel_test(parameter_tuple):
    best_model = get_model_path(parameter_tuple)
    gpu_id = current_process()._identity[0] - 1

    parameter_dict = decode_parameter_tuple(parameter_tuple)

    scores = []

    # test set evaluation
    for te in parameter_dict['te']:
        logger.info(f'''evaluating {te}''')
        eval_out_path = os.path.join(best_model, 'te_predictions.csv')
        classification_report_path = os.path.join(best_model, 'te_classification_report.txt')
        report_path = os.path.join(best_model, 'full_report.txt')

        LABEL_COLUMN_NAME = parameter_dict['LABEL_COLUMN_NAME']
        logger.info('making predictions')
        print('TEST FILE')
        print(te)
        cmd = ['python', 'evaluate.py',
               '--gpu_id', str(gpu_id),
               '-t', os.path.abspath(te),
               '-s', os.path.abspath(best_model),
               '-o', eval_out_path,
               '-m', 'get_pred']
        logger.info(' '.join(cmd))
        subprocess.run(cmd)

        logger.info('generating classification report')
        df = read_csv_json(eval_out_path)
        ground_truths = [x for x in eval(f'df.{LABEL_COLUMN_NAME}')]
        #print(ground_truths)
        ground_truth_groups = []
        for group in eval(f'df.{LABEL_COLUMN_NAME}'):
            if type(group) == str:
                labels = group.split('|')
            else:
                labels = []
            ground_truth_groups.append(labels)
        ground_truths = MultiLabelBinarizer().fit_transform(ground_truth_groups)
        predicted_labels = [x for x in df.elmo_pred_intent]
        predicted_label_groups = []
        for group in predicted_labels:
            if type(group) == str:
                labels = group.split('|')
            else:
                labels = []
            predicted_label_groups.append(labels)
        print('predicted label groups')
        print(predicted_label_groups)
        predicted_labels = MultiLabelBinarizer().fit_transform(predicted_label_groups)
        print('predicted labels')
        print(predicted_labels)
        pred_scores = list(df.elmo_pred_conf)

        # if starter packs dataset, then determine the optimal threshold using test set,
        # and then threshold the predictions
        if parameter_dict['data'] in STARTER_PACKS_DATA:
            if parameter_dict['encoder'] != 'fasttext-wiki':  # use va set to determine the threshold, and use te to

                # evluation va_threshold #####################################
                logger.info(f'''evaluating {parameter_dict['va_threshold']}''')
                va_threshold_eval_out_path = os.path.join(best_model, 'va_threshold_predictions.csv')
    
                logger.info('making predictions')
                cmd = ['python', 'evaluate.py',
                       '--gpu_id', str(gpu_id),
                       '-t', os.path.abspath(parameter_dict['va_threshold']),
                       '-s', os.path.abspath(best_model),
                       '-o', va_threshold_eval_out_path,
                       '-m', 'get_pred']
                logger.info(' '.join(cmd))
                subprocess.run(cmd)

                logger.info('generating classification report')
                va_threshold_df = read_csv_json(va_threshold_eval_out_path)
                va_threshold_ground_truths = [x for x in eval(f'va_threshold_df.{LABEL_COLUMN_NAME}')]
                va_threshold_predicted_labels = [x for x in va_threshold_df.elmo_pred_intent]
                va_threshold_pred_scores = list(va_threshold_df.elmo_pred_conf)
                ##############################################################

                threshold_range = numpy.round(numpy.arange(0.1, 1.001, 0.001), decimals=5)
                best_threshold = 0.0
                f1_macro_under_best_threshold = 0.0
                for threshold_index, threshold in enumerate(threshold_range):
                    va_threshold_predicted_labels_copy = deepcopy(va_threshold_predicted_labels)
                    # print('+++++', len(va_threshold_predicted_labels_copy), len(va_threshold_pred_scores), len(va_threshold_df))
                    for i, score in enumerate(va_threshold_pred_scores):
                        if score < threshold:
                            va_threshold_predicted_labels_copy[i] = 'undefined'
                    f1_macro_this_threshold = f1_score(y_true=va_threshold_ground_truths, y_pred=va_threshold_predicted_labels_copy, average='macro')
                    if f1_macro_this_threshold > f1_macro_under_best_threshold:
                        best_threshold = threshold
                        f1_macro_under_best_threshold = f1_macro_this_threshold
                logger.info(f'best_threshold decided by va_threshold set: {best_threshold}')

                # now threshold the predictions under the best threshold
                for i, score in enumerate(pred_scores):
                    if score < best_threshold:
                        predicted_labels[i] = 'undefined'
            else:
                fixed_threshold = 0.7

                logger.info(f'fixed_threshold: {fixed_threshold}')

                # now threshold the predictions under the best threshold
                for i, score in enumerate(pred_scores):
                    if score < fixed_threshold:
                        predicted_labels[i] = 'undefined'
  
        classification_report_this_fold = classification_report(y_true=ground_truths, y_pred=predicted_labels)
        f1_macro_this_fold = f1_score(y_true=ground_truths, y_pred=predicted_labels, average='macro')
        precision_macro_this_fold = precision_score(y_true=ground_truths, y_pred=predicted_labels, average='macro')
        recall_macro_this_fold = recall_score(y_true=ground_truths, y_pred=predicted_labels, average='macro')
        acc_this_fold = accuracy_score(y_true=ground_truths, y_pred=predicted_labels)
        with open(classification_report_path, 'wt') as f:
            f.write(classification_report_this_fold)

        logger.info(f'classification report:\n{classification_report_this_fold}')
        logger.info('*' * 50 + ' ' + 'te' + ' ' + '*' * 50)
        logger.info(f'classification report output path: {classification_report_path}')
        logger.info(f'size: {len(ground_truths)}')
        logger.info(f'acc score: {round(acc_this_fold, 3)}')
        logger.info(f'f1 macro score: {round(f1_macro_this_fold, 3)}')
        return parameter_tuple[:len(separate_models)], (f1_macro_this_fold, precision_macro_this_fold, recall_macro_this_fold)


def main(args, logger):
#    os.environ['TORCH_MODEL_ZOO'] = '/home/jovyan/embeddings/deep-sentence-classifiers/tmp/'

    #pool = Pool(4)  # 4 GPUs
    pool = Pool(4)
    if not args.disable_training:
        va_score_dict = dict(pool.map(parallel_train_eval, parameter_grid_tuple_type))

        # select the best model from the trials, using va set performance
        va_score_dict_separate_models = defaultdict(
            list)  # a dict: key is the separate_model parameters, values are (parameters_tuple, va_performance) tuple
        for k, v in va_score_dict.items():
            va_score_dict_separate_models[k[:len(separate_models)]].append((k, v))

        # now, find the best model    # select the best model from the trials, using va set performance
        best_model_dict = {}  # a dict: key is the separate_model parameters, values are best model paths
        for k, v in va_score_dict_separate_models.items():
            best_model_tuple, _ = max(v, key=operator.itemgetter(1))
            best_model_dict[k] = best_model_tuple
            logger.info(f'best model for {k} is {best_model_dict[k]}')
        print("BEST_MODEL_DICT_PATH")
        print(BEST_MODEL_DICT_PATH)
        with open(BEST_MODEL_DICT_PATH, 'wb') as f:
            pickle.dump(best_model_dict, f)
    else:
        with open(BEST_MODEL_DICT_PATH, 'rb') as f:
            best_model_dict = pickle.load(f)
            logger.info(f'loaded the best model dict {best_model_dict}')

    print("DONE TRAINING")
    # test models
    te_score_dict = dict(pool.map(parallel_test, best_model_dict.values()))
    logger.info('te_score_dict')
    logger.info(str(te_score_dict))
    print("DONE TESTING")

    # make a pandas dataframe
    separate_model_parameters = list(separate_models.keys())
    te_score_df = pandas.DataFrame(columns=separate_model_parameters + ['test f1 macro score'] + ['test precision macro score'] + ['test recall macro score'])
    print('TE SCORE DICTIONARY')
    print(te_score_dict)
    for k, v in te_score_dict.items():
        append_dict = {**dict(zip(separate_model_parameters, k)), 'test f1 macro score': v}
        te_score_df = te_score_df.append(append_dict, ignore_index=True)

    data_combined = '_'.join(separate_models['data'])
    te_score_df.to_csv(os.path.join(args.out_path, f'te_scores_{data_combined}.csv'))


if __name__ == '__main__':
#    os.environ['TORCH_MODEL_ZOO'] = '/home/jovyan/embeddings/deep-sentence-classifiers/tmp/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_training', action='store_true')
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--out_path', type=str, default='output_encoders')
    parser.add_argument('--encoder', type=str, nargs='+',
                        help='select one or more encoder from m-bert, distil-m-bert, xlm-roberta, elmo-small-lp, fasttext-wiki, bert, roberta, 11-brand-distil-bert, distil-bert, albert, elmo, distil-m-bert-eng-esp, distil-m-bert-esp, distil-m-bert-por, distil-m-bert-eng-esp-por')
    parser.add_argument('--classifier', type=str, nargs='+',
                        help='select one or more classifier from cnn, lp_nlu_att, lr, mapxpooling')
    parser.add_argument('--embedding_type', type=str, nargs='+', default='word',
                        help='select one or more embedding type from word, max, mean, cls, first+last')
    parser.add_argument('--data', type=str, nargs='+',
                        help='select one or more dataset from usic, CenturyLink, Hawaiian, Telstra')
    parser.add_argument('--language', type=str, nargs='+',
                        help='select one or more language from eng, esp, por, eng-esp, esp-eng, xlt-eng-esp, xlt-eng-por')
    args = parser.parse_args()

    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)

    ################### mutable code starts here
    BEST_MODEL_DICT_PATH = os.path.join(args.out_path, 'best_model_dict.pkl')

    # grid search hyperparameters.
    # separate_models ar:e the parameters that's reported differently
    # separate_runs are the parameters to choose the best model from
    #separate_models = [('encoder', ['distil-m-bert-fine-tuned']),
    #separate_models = [('encoder', ['11-brand-distil-m-bert', 'distil-m-bert-eng-esp', 'distil-m-bert-por']),
    #separate_models = [('encoder', ['11-brand-distil-m-bert', 'distil-m-bert-esp','distil-m-bert-eng-esp']),
    separate_models = [('encoder', args.encoder),
    #separate_models = [('encoder', ['11-brand-distil-bert', '11-brand-distil-m-bert', 'distil-bert', 'distil-m-bert', 'elmo-small-lp']),# 'bert', 'roberta', 'm-bert', 'distil-m-bert', 'xlm-roberta', 'elmo-small-lp', 'fasttext-wiki']),  # 'bert', 'roberta', '11-brand-distil-bert', 'distil-bert', 'albert', 'elmo'
                       ('classifier', args.classifier),  # ['lp_nlu_att', 'lr', 'cnn', 'multi_att']
                       ('embedding_type', args.embedding_type),
                       ('data', args.data),  # , 'usic', 'CenturyLink', 'Hawaiian', 'Telstra'
                       #('data', ['usic']),  # , 'usic', 'CenturyLink', 'Hawaiian', 'Telstra'
                       ('language', args.language)]
    #                   ('language', ['xlt-eng-por','por'])]
    #                   ('language', ['eng', 'esp', 'eng-esp', 'esp-eng'])]
    separate_runs = [('run', list(range(args.n_trials)))]
    ################### mutable code ends here

    all_hyperparameters = separate_models + separate_runs
    separate_models = OrderedDict(separate_models)
    separate_runs = OrderedDict(separate_runs)
    all_hyperparameters = OrderedDict(all_hyperparameters)

    parameter_grid = list(ParameterGrid(all_hyperparameters))

    parameter_grid_tuple_type = [tuple((parameter_dict[x] for x in all_hyperparameters.keys())) for parameter_dict in
                                 parameter_grid]

    parameter_grid_separate_models = list(ParameterGrid(separate_models))
    parameter_grid_separate_models_tuple_type = [tuple((parameter_dict[x] for x in separate_models.keys())) for
                                                 parameter_dict in
                                                 parameter_grid_separate_models]

    # start a logger
    format_str = '[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(level=logging.INFO, format=format_str)
    logger = logging.getLogger(__name__)
    log_path = os.path.join(args.out_path, 'benchmark_architecture_log.txt')
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)

    main(args, logger)
