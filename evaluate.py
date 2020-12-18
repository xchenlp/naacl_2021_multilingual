import argparse
import os
from copy import deepcopy
from typing import Union

#import matplotlib
#import matplotlib.pyplot as plt
import numpy
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from utils.loader import Loader
from models import SentenceClassificationModel


#matplotlib.use('agg')


def evaluate(data_loader, device, model, set_name, args):
    model.eval()
    pred_labels = []
    confs_list = []

    for step in range(data_loader.total_batch):
        message_tensor, label_tensor, attention_mask = data_loader.next_batch()
        message_tensor = message_tensor.to(device)
        attention_mask = attention_mask.to(
            device) if attention_mask is not None else None

        # run
        pred, confidence_score = model(message_tensor, attention_mask=attention_mask)
        confs = torch.nn.functional.softmax(confidence_score, dim=1).cpu().data.numpy()
        max_confs = numpy.amax(confs, 1)
        #pred_labels.extend([ind2cls[int(x)] for y in pred.cpu().data.numpy() for x in y])
       
        for binary_values in pred.cpu().data.numpy():
            labels = []
            i = 0
            while i < len(binary_values):
                if int(binary_values[i]) == 1:
                    labels.append(ind2cls[i])
                i += 1
            pred_labels.append('|'.join(sorted(labels)))
             
        #for group in pred.cpu().data.numpy():
        #    pred_labels.append('|'.join([ind2cls[int(label)] for label in group].sort()))
        
        confs_list.extend([float(x) for x in max_confs])
        print(f'{step} of {data_loader.total_batch} finished')

    output_df = deepcopy(data_loader.df)
    print(len(output_df), len(pred_labels), len(confs_list))
    output_df = output_df.iloc[:meta_args.datacap]
    output_df['elmo_pred_intent'] = pred_labels
    output_df['elmo_pred_conf'] = confs_list
    if meta_args.out_path is None:
        output_df.to_csv(f'{meta_args.test_set_path}.predictions.csv')
    else:
        output_df.to_csv(meta_args.out_path)


def evaluate_top_n(data_loader, device, model, set_name, args):
    model.eval()
    n = len(cls2ind)
    pred_labels = numpy.zeros((len(data_loader.data), n), dtype=int)
    confs_list = numpy.zeros((len(data_loader.data), n), dtype=float)
    # pred_labels_sanity = []

    for step in range(data_loader.total_batch):
        message_tensor, label_tensor, attention_mask = data_loader.next_batch()
        message_tensor = message_tensor.to(device)
        attention_mask = attention_mask.to(
            device) if attention_mask is not None else None

        # run
        pred, confidence_score = model(
            message_tensor, attention_mask=attention_mask)

        confs = torch.nn.functional.softmax(
            confidence_score, dim=1).cpu().data.numpy()
        top_n_labels = (confs.argsort(axis=1)[:, ::-1])[:, :n]
        # pred_labels_sanity.extend([ind2cls[int(x)] for x in pred.cpu().data.numpy()])
        for i in range(confs.shape[0]):
            global_i = data_loader.batch_size * step + i
            pred_labels[global_i, :] = top_n_labels[i]
            confs_list[global_i, :] = [
                float(confs[i, x]) for x in top_n_labels[i]]

        print(f'{step} of {data_loader.total_batch} finished')
    output_df = pd.read_csv(meta_args.test_set_path, lineterminator='\n')
    output_df = output_df.iloc[:meta_args.datacap]
    for n_th in range(n):
        output_df[f'elmo_pred_intent_top_{n_th + 1}'] = [
            ind2cls[int(x)] for x in pred_labels[:, n_th]]
        output_df[f'elmo_pred_conf_top_{n_th + 1}'] = confs_list[:, n_th]
        # set_trace()
    output_df.to_csv(f'{meta_args.test_set_path}.top_n_predictions.csv')


def plot_confidence_score_distribution_histogram(confidence_scores: Union[list, numpy.ndarray],
                                                 file_name: str = 'confidence_score_distribution_histogram'):
    fig, ax = plt.subplots()
    plt.hist(confidence_scores, bins=100)
    plt.xlabel('confidence score')
    plt.ylabel('count')
    plt.title("confidence score distribution_histogram")
    fig.savefig(f'{file_name}.png')


def evaluate_with_ground_truth(data_loader, device, model, set_name, args):
    model.eval()
    true_labels = []
    pred_labels = []
    sic_confidence = []

    for step in range(data_loader.total_batch):
        message_tensor, label_tensor, attention_mask = data_loader.next_batch()
        message_tensor = message_tensor.to(device)
        attention_mask = attention_mask.to(
            device) if attention_mask is not None else None
        label_tensor = label_tensor.to(device)
        if args.label == 'soft' and set_name == 'tr':
            _, label_tensor = torch.max(label_tensor, 1)
        # run
        pred, confidence_score = model(
            message_tensor, attention_mask=attention_mask)

        confs = torch.nn.functional.softmax(
            confidence_score, dim=1).cpu().data.numpy()
        sic_confs = confs[:, 0]
        max_confs = numpy.amax(confs, 1)

        # If the confidence score of the prediction < self.model_confidence_threshold
        # change the prediction to `not_intent`
        replace_with_other = max_confs <= args.model_confidence_threshold
        for i, should_be_replaced in enumerate(replace_with_other):
            if should_be_replaced:
                pred[i] = cls2ind['other']

        true_labels.extend([int(x) for x in label_tensor.cpu().data.numpy()])
        pred_labels.extend([int(x) for x in pred.cpu().data.numpy()])
        sic_confidence.extend([float(x) for x in sic_confs])

    # flatten list for multiple labels per message to list of labels, to calculate f1 and acc
    true_labels_flattened = [x for y in true_labels for x in y]
    pred_labels_flattened = [x for y in pred_labels for x in y]

    f1_macro = f1_score(true_labels_flattened, pred_labels_flattened, average='macro')
    precision_macro = precision_score(true_labels_flattened, pred_labels_flattened, average='macro')
    recall_macro = recall_score(true_labels_flattened, pred_labels_flattened, average='macro')
    acc = accuracy_score(true_labels_flattened, pred_labels_flattened)

    for i in range(len(data_loader.data)):
        print(data_loader.data[i][0])
        print('true:', true_labels[i])
        print('pred:', pred_labels[i])
        print('sic_conf:', sic_confidence[i])
        print()
    print(cls2ind)

    print('*' * 50)

    function_lib = {'f1_score': lambda x, y: f1_score(true_labels_flattened, pred_labels_flattened, average='macro'),
                    'precision_score': lambda x, y: precision_score(true_labels_flattened, pred_labels_flattened, average='macro'),
                    'recall_score': lambda x, y: recall_score(true_labels_flattened, pred_labels_flattened, average='macro'),
                    'accuracy_score': accuracy_score}
    for function_name, score_function in function_lib.items():
        score = score_function(true_labels_flattened, pred_labels_flattened)
        # conf_interval_ = get_conf_interval(true_labels, pred_labels, metric_f=score_function)
        # conf_per_class_ = conf_per_class(true_labels, pred_labels, metric_f=score_function)
        print(f'{set_name} set {function_name}: {score}.')
        # print(f'conf_interval_: {conf_interval_}.')
        # print(f'conf_per_class_: {conf_per_class_}.')
    # classification report

    labels = list(cls2ind.values())
    target_names = list(cls2ind.keys())

    print(classification_report(true_labels_flattened, pred_labels_flattened, labels=labels, target_names=target_names))

    # confidence sore distribution for sic sentences (< 0.5 conf means wrong classification)
    is_sic_scores = []
    not_sic_scores = []
    for i in range(len(true_labels)):
        if true_labels[i] == 0:  # is sic
            is_sic_scores.append(sic_confidence[i])
        else:
            not_sic_scores.append(sic_confidence[i])
    plot_confidence_score_distribution_histogram(
        is_sic_scores, 'is_sic_scores')
    plot_confidence_score_distribution_histogram(
        not_sic_scores, 'not_sic_scores')

    return true_labels, pred_labels, acc, f1_macro, precision_macro, recall_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a deep intent classification model.')
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='''Usually the timestamp folder location, such as '/data/deep-sentence-classifiers-data/2019_06_07_13_54_27_060955'.''')
    parser.add_argument('-t', '--test_set_path', type=str, nargs='+',
                        default='data/val_travel_sic_dedupe.jsonl.remove_ignore.json',
                        help='test set location')
    parser.add_argument('-o', '--out_path', type=str)
    parser.add_argument('--batch_size', type=int,
                        default=128, help='test set batch size')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='test set batch size')
    parser.add_argument('--datacap', type=int,
                        default=100000000000, help='test set batch size')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='get_metrics, get_top_n_pred, or get_pred')
    meta_args = parser.parse_args()

    print(meta_args)

    args_save_name = os.path.join(meta_args.save_dir, 'args.pt')
    args = torch.load(args_save_name)

    # initialize path variables, logging, and database, and the GPU/ CPU to use
    device = torch.device(f"cuda:{meta_args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print('DEVICE')
    print(device)
    # generate intent class dict
    intent_list_save_name = os.path.join(meta_args.save_dir, 'intent_list.pt')
    intent_list = torch.load(intent_list_save_name)
    cls2ind = {v: i for i, v in enumerate(intent_list)}
    ind2cls = {v: k for k, v in cls2ind.items()}
    print(ind2cls)
    cls_num = len(cls2ind)

    # generate the deep intent model
    model = SentenceClassificationModel(intent_list, args)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # load pretrained model
    model_save_name = os.path.join(meta_args.save_dir, 'model.pt')
    #quick hack for comparing pretrained models
    if 'classifier' in meta_args.save_dir:
        #model.load_state_dict(torch.load('output_encoders/distil-m-bert-eng-esp_cnn_usic_esp_0/model.pt', map_location=device))
        model.classifier.load_state_dict(torch.load(model_save_name))
    else:
        print(f'loading model {model_save_name}')
        model.load_state_dict(torch.load(model_save_name))


    # now, run the evaluation I want and quit
    # remember I can add some flags to the evaluation function
    # so the eval does something different from ordinary eval
    if meta_args.mode in ['get_top_n_pred', 'get_pred']:
        label = 'none'
    elif meta_args.mode == 'get_metrics':
        label = 'hard'
    else:
        raise NotImplementedError

    # test set is always hard labeled
    test_data_loader = Loader(meta_args.test_set_path, cls2ind, batch_size=meta_args.batch_size, shuffle=False,
                              label=label, eval=True, num_total_epochs=1 + 5, datacap=meta_args.datacap, args=args)
    if meta_args.mode == 'get_pred':
        evaluate(test_data_loader, device, model, 'test', args)
    elif meta_args.mode == 'get_metrics':
        evaluate_with_ground_truth(
            test_data_loader, device, model, 'test', args)
    elif meta_args.mode == 'get_top_n_pred':
        evaluate_top_n(test_data_loader, device, model, 'test', args)
