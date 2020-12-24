import argparse
import datetime
import logging
import os
from sklearn.metrics import classification_report
import pickle
import time
import yaml

#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import tensor
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert import BertTokenizer
from transformers import XLMRobertaTokenizer, DistilBertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from utils.loader import Loader
from models import SentenceClassificationModel
from utils.sanity_check import sanity_check
from utils.sql_writer import SQLWriter
from torch.optim import Adam
from pdb import set_trace


#matplotlib.use('agg')
os.environ['TORCH_MODEL_ZOO'] = '/home/jovyan/embeddings/deep-sentence-classifiers/tmp/'

def initialize(args):
    timestamp = '{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.datetime.now())

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    # starting logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info(f'log timestamp: {timestamp}')
    logger.info(args)

    # model save place
    model_save_name = os.path.join(args.save_dir, 'model.pt')
    va_labels_save_name = os.path.join(args.save_dir, 'validation_labels')
    intent_list_save_name = os.path.join(args.save_dir, 'intent_list.pt')

    args_save_name = os.path.join(args.save_dir, 'args.pt')
    torch.save(args, args_save_name)

    return logger, model_save_name, va_labels_save_name, intent_list_save_name, timestamp


def evaluate(data_loader, device, model, set_name, logger, args, mode=None):
    model.eval()
    true_labels = []
    pred_labels = []
    if mode == 'plot_prob_distr':
        highest_prob = np.zeros((data_loader.data_size,))
    elif mode == 'plot_single_scores_other_class':
        other_counter = 0

    for step in range(data_loader.total_batch):
        message_tensor, label_tensor, attention_mask = data_loader.next_batch()
        message_tensor = message_tensor.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        label_tensor = label_tensor.to(device)
        if args.label == 'soft' and set_name == 'tr':
            _, label_tensor = torch.max(label_tensor, 1)
        # run
        pred, confidence_score = model(
            message_tensor, attention_mask=attention_mask)

        # If the confidence score of the prediction < self.model_confidence_threshold
        # change the prediction to `not_intent`
        confs = torch.nn.functional.softmax(
            confidence_score, dim=1).cpu().data.numpy()
        max_confs = np.amax(confs, 1)
        replace_with_other = max_confs <= args.model_confidence_threshold
        for i, should_be_replaced in enumerate(replace_with_other):
            if should_be_replaced:
                pred[i] = cls2ind['other']
        true_labels.extend([x for x in label_tensor.cpu().data.numpy()])
        pred_labels.extend([x for x in pred.cpu().data.numpy()])

        if mode == 'plot_single_scores':
            if model.encoder_name == 'm-bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            elif 'distil-m-bert' in model.encoder_name:   
            #elif model.encoder_name in ['distil-m-bert', '11-brand-distil-m-bert', 'distil-m-bert-eng-esp', 'distil-m-bert-eng-esp-por', 'distil-m-bert-por', 'distil-m-bert-esp', 'distil-m-bert-fine-tuned']:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
            elif model.encoder_name in ['distil-bert', '11-brand-distil-bert']:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            elif model.encoder_name == 'xlm-roberta':
                tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            else:#khowell I'm not sure why just berttokenizer was used origionally. there might be other models we need to change htis for
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # output the confidence scores and the message
            for i in range(args.batch_size):
                conf_i = torch.nn.functional.softmax(confidence_score[i], dim=0).cpu().data.numpy()
                message = ' '.join(tokenizer.convert_ids_to_tokens(message_tensor[i].cpu().data.numpy())[:18])
                plt.rcParams.update({'font.size': 5})
                fig, ax = plt.subplots()
                x_labels = list(range(cls_num))
                plt.bar(x_labels, conf_i)
                plt.xticks(x_labels, [ind2cls[x]
                                      for x in x_labels], rotation='vertical')
                plt.xlabel('class')
                plt.ylabel('probability')
                plt.title(f'confidence scores: {message}')
                # plt.legend()
                fig.tight_layout()
                fig.savefig(f'figures/{i}.png', dpi=1000)
                plt.close()
            exit()
        if mode == 'plot_single_scores_other_class':
            if model.encoder_name == 'm-bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            elif 'distil-m-bert' in model.encoder_name:
            #elif model.encoder_name in ['distil-m-bert', '11-brand-distil-m-bert', 'distil-m-bert-eng-esp', 'distil-m-bert-eng-esp-por', 'distil-m-bert-por', 'distil-m-bert-esp', 'distil-m-bert-fine-tuned']:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
            elif model.encoder_name in ['distil-bert', '11-brand-distil-bert']:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            elif model.encoder_name == 'xlm-roberta':
                tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            else:#khowell I'm not sure why just berttokenizer was used origionally. there might be other models we need to change htis for
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # output the confidence scores and the message
            for i in range(args.batch_size):
                conf_i = torch.nn.functional.softmax(
                    confidence_score[i], dim=0).cpu().data.numpy()
                cls_name = ind2cls[np.argmax(conf_i, axis=0)]
                if cls_name == 'other':
                    message = ' '.join(tokenizer.convert_ids_to_tokens(
                        message_tensor[i].cpu().data.numpy())[:18])
                    plt.rcParams.update({'font.size': 5})
                    fig, ax = plt.subplots()
                    x_labels = list(range(cls_num))
                    plt.bar(x_labels, conf_i)
                    plt.xticks(x_labels, [ind2cls[x]
                                          for x in x_labels], rotation='vertical')
                    plt.xlabel('class')
                    plt.ylabel('probability')
                    plt.title(f'confidence scores: {message}')
                    # plt.legend()
                    fig.tight_layout()
                    fig.savefig(f'figures/other_{other_counter}.png', dpi=1000)
                    plt.close()
                    other_counter += 1
                    if other_counter >= 40:
                        exit()
        elif mode == 'plot_prob_distr':
            confidence_score = torch.nn.functional.softmax(confidence_score, dim=1).cpu().data.numpy()
            highest_prob[step * args.batch_size: (step + 1) *
                                                 args.batch_size] = np.max(confidence_score, axis=1)
    if mode == 'plot_prob_distr':
        fig, ax = plt.subplots()
        plt.hist(highest_prob, bins=40, range=(0, 1))
        plt.xlabel('confidence score')
        plt.ylabel('count of most confident answers')
        plt.title('confidence score distribution')
        fig.savefig('figures/confidence_score_distribution.png', dpi=700)
        plt.close()

    # flatten list for multiple labels per message to list of labels, to calculate f1 and acc
    true_labels_flattened = [x for y in true_labels for x in y]
    pred_labels_flattened = [x for y in pred_labels for x in y]

    f1_macro = f1_score(true_labels_flattened, pred_labels_flattened, average='macro')
    acc = accuracy_score(true_labels_flattened, pred_labels_flattened)

    # function_lib = {'f1_score': lambda x, y: f1_score(true_labels, pred_labels, average='macro'),
    #                 'precision_score': lambda x, y: precision_score(true_labels, pred_labels, average='macro'),
    #                 'recall_score': lambda x, y: recall_score(true_labels, pred_labels, average='macro'),
    #                 'accuracy_score': accuracy_score}
    # for function_name, score_function in function_lib.items():
    #     score = score_function(true_labels, pred_labels)
    #     # conf_interval_ = get_conf_interval(true_labels, pred_labels, metric_f=score_function)
    #     # conf_per_class_ = conf_per_class(true_labels, pred_labels, metric_f=score_function)
    #     logger.info(f'{set_name} set {function_name}: {score}.')
    #     # logger.info(f'conf_interval_: {conf_interval_}.')
    #     # logger.info(f'conf_per_class_: {conf_per_class_}.')
    return true_labels, pred_labels, acc, f1_macro


def evaluate_soft(data_loader, device, model, set_name, logger, args):
    model.eval()
    loss_sum = 0

    for step in range(data_loader.total_batch):
        message_tensor, label_tensor, attention_mask = data_loader.next_batch()
        message_tensor = message_tensor.to(device)
        attention_mask = attention_mask.to(
            device) if attention_mask is not None else None
        label_tensor = label_tensor.to(device)

        # run
        loss, pred, _ = model(
            message_tensor, attention_mask=attention_mask, ground_truth_labels=label_tensor)
        loss_sum += float(loss.cpu().data.numpy())
    # TODO: This is a hack. Change this hack. Treat 100000 - loss_sum as the optimization
    acc = 0
    f1_macro = 100000 - loss_sum
    true_labels = None
    pred_labels = None

    return true_labels, pred_labels, acc, f1_macro


def train(model, tr_data_path, va_data_path, cls2ind, batch_size, logger, db_writer, optimizer, device, args):
    os.environ['TORCH_MODEL_ZOO'] = '/home/jovyan/embeddings/deep-sentence-classifiers/tmp/'

    # start data loader
    tr_data_loader = Loader(tr_data_path, cls2ind, batch_size=batch_size, shuffle=True,
                            datacap=args.datacap, label=args.label, soft_marginals=args.soft_marginals,
                            num_total_epochs=args.epoch + 5, first_time_inspection=args.first_time_inspection,
                            args=args, eval=True)  # TODO: understand join and daemon
    if not args.no_evaluation:
        va_data_loader = Loader(va_data_path, cls2ind, batch_size=batch_size, shuffle=False,
                                label='soft' if args.soft_validation_set else 'hard',
                                soft_marginals=args.va_soft_marginals, eval=True, num_total_epochs=args.epoch + 5,
                                args=args)  # TODO: understand join and daemon

    # texts = [len(x[0].strip()) for x in self.data]

    # do the sanity checks: lowercase and duplicates in va and tr
    if not args.override_sanity_check:
        if args.no_evaluation:
            pass
        else:
            sanity_check([text for text, _ in tr_data_loader.data], [text for text, _ in va_data_loader.data], format='list')
        print('sanity check passed')

    # initialize the recording variables
    logger.info(f'tr size: {tr_data_loader.data_size}. va size: {va_data_loader.data_size if not args.no_evaluation else 0}')
    best_tr_acc = 0
    best_tr_f1_macro = 0
    best_tr_epoch = 0
    best_va_acc = 0
    best_va_f1_macro = 0
    best_va_epoch = 0

    # start the training epochs
    for epoch in range(1, args.epoch + 1):
        logger.info(f'Epoch {epoch} started')
        # set model to training mode
        model.train()
        start = time.time()
        for step in range(tr_data_loader.total_batch):
            # get data from the datapump queues
            message_tensor, label_tensor, attention_mask = tr_data_loader.next_batch()

            # forward
            optimizer.zero_grad()
            message_tensor = message_tensor.to(device)
            attention_mask = attention_mask.to(
                device) if attention_mask is not None else None
            label_tensor = label_tensor.to(device)

            loss, pred, _ = model(message_tensor, attention_mask=attention_mask, ground_truth_labels=label_tensor)

            # backward
            if args.multi_gpu and torch.cuda.device_count() > 1:
                loss.mean().backward()
            else:
                loss.backward()
            optimizer.step()

            # log predictions
            if args.label == 'soft':
                _, label_tensor = torch.max(label_tensor, 1)
            label_tensor = label_tensor == 1
            correct_minibatch = 0
            i = 0
            while i < len(label_tensor):
               if np.array_equal(label_tensor.cpu().data.numpy()[i], pred.cpu().data.numpy()[i]):
                    correct_minibatch += 1
               i += 1
            #correct_minibatch = int(
            #    sum(pred.cpu().data.numpy() == label_tensor.cpu().data.numpy()))
            if step % args.report_interval == 0:
                logger.info(f'step: {tr_data_loader.batch_idx}')
                logger.info(
                    f'loss: {loss}. correct percentage: {int(correct_minibatch / tr_data_loader.batch_size * 100)}%')
                logger.info(f'pred: {pred[:9]}')
                logger.info(f'true: {label_tensor[:9]}')

        end = time.time()
        tr_time = end - start
        logger.info(f'Training time: {tr_time}')

        if args.no_evaluation:
            va_time = 0.0
            va_acc = 0.0
            va_f1_macro = 0.0
        else:
            # evaluate this epoch of training
            if args.with_tr_report:
                _, _, tr_acc, tr_f1_macro = evaluate(tr_data_loader, device, model, 'tr', logger, args)
            start = time.time()
            if args.soft_validation_set:
                va_true_labels, va_pred_labels, va_acc, va_f1_macro = \
                    evaluate_soft(va_data_loader, device, model, 'va', logger, args)
            else:
                va_true_labels, va_pred_labels, va_acc, va_f1_macro = \
                    evaluate(va_data_loader, device, model, 'va', logger, args)
            end = time.time()
            va_time = end - start
            logger.info(f'Evaluation time: {va_time}')
            if args.with_tr_report:
                if tr_acc >= best_tr_acc:
                    best_tr_acc = tr_acc
                    best_tr_epoch = epoch
                    best_tr_f1_macro = tr_f1_macro

            if epoch == 1 and not args.soft_validation_set:
                # save the true labels
                true_labels = []
                for binary_values in va_true_labels:
                    labels = []
                    i = 0
                    while i < len(binary_values):
                        if int(binary_values[i]) == 1:
                            labels.append(ind2cls[i])
                        i += 1
                    true_labels.append('|'.join(sorted(labels)))
                    #true_labels.append('|'.join([ind2cls[label] for label in group].sort()))
                torch.save(true_labels, va_labels_save_name + '.true.pt')
                #torch.save([ind2cls[x] for y in va_true_labels for x in y], va_labels_save_name + '.true.pt')

        if args.no_evaluation or va_f1_macro > best_va_f1_macro:
            best_va_acc = va_acc
            best_va_epoch = epoch
            best_va_f1_macro = va_f1_macro
            # save the best model
            torch.save(model.state_dict(), model_save_name)
            logger.info(f'saving model at epoch {epoch}')
            # save the best predictions

            if not args.no_evaluation and not args.soft_validation_set:
                pred_labels = []
                for binary_values in va_pred_labels:
                    labels = []
                    i = 0
                    while i < len(binary_values):
                        if int(binary_values[i]) == 1:
                            labels.append(ind2cls[i])
                        i += 1
                    pred_labels.append('|'.join(sorted(labels)))
                torch.save(pred_labels, va_labels_save_name + '.pred.pt')
                #torch.save(pred_labels, va_labels_save_name + '.pred.pt')
                #torch.save([ind2cls[x] for y in va_pred_labels for x in y], va_labels_save_name + '.pred.pt')
        db_writer.write_entry(timestamp, str(args), best_tr_f1_macro, best_tr_acc, best_tr_epoch,
                              best_va_f1_macro, best_va_acc, best_va_epoch, epoch, tr_time, va_time, False)

        # measure whether to early stop
        if epoch - best_va_epoch == args.early_stopping:
            logger.info('early stopping')
            break
        logger.info(f'epoch {epoch} has finished.')
        logger.info(f'Best epoch is {best_va_epoch} with va acc {best_va_acc} and va f1-macro: {best_va_f1_macro}.')

    # after training we generate the classification report
    #db_writer.write_entry(timestamp, str(args), best_tr_f1_macro, best_tr_acc, best_tr_epoch,
    #                      best_va_f1_macro, best_va_acc, best_va_epoch, epoch, tr_time, va_time, True)
    if not args.soft_validation_set and not args.no_evaluation:
        true = torch.load(os.path.join(args.save_dir, 'validation_labels.true.pt'))
        pred = torch.load(os.path.join(args.save_dir, 'validation_labels.pred.pt'))
        logger.info('best model performance on va set')
        logger.info(classification_report(y_true=true, y_pred=pred))
    else:
        logger.info('No classification report generated.')


def get_optimizer(model, lr, weight_decay, optimizer_name):
    if optimizer_name == 'bert-adam':
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr)
    elif optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def generate_intent_list(args, intent_list_save_name):
    if args.label == 'hard':
        cls2ind = {}
        df = pd.DataFrame()
        print('list')
        print(args.tr)
        for file in args.tr:
            print('file')
            print(file)
            if file.endswith('csv'):
                df = df.append(pd.read_csv(file, lineterminator='\n'))
            elif file.endswith('json') or file.endswith('jsonl'):
                df = df.append(pd.read_json(file, lines=True))
            else:
                raise NotImplementedError('not json or csv format')
        if 'multi' in args.classifier:
            for row in df.itertuples():
                labels = eval(f'row.{args.label_column_name}').split('|')
                for label_class in labels:
                    if label_class not in cls2ind:
                        if not args.remove_other:
                            cls2ind[label_class] = len(cls2ind)
                        elif args.remove_other and label_class != 'other' and label_class != 'Other':
                            cls2ind[label_class] = len(cls2ind)
            ind2cls = {v: k for k, v in cls2ind.items()}
            intent_list = [ind2cls[i] for i in range(len(ind2cls))]
            torch.save(intent_list, intent_list_save_name)

        else:
            for row in df.itertuples():
                label_class = eval(f'row.{args.label_column_name}')
                if label_class not in cls2ind:
                    if not args.remove_other:
                        cls2ind[label_class] = len(cls2ind)
                    elif args.remove_other and label_class != 'other' and label_class != 'Other':
                        cls2ind[label_class] = len(cls2ind)

            ind2cls = {v: k for k, v in cls2ind.items()}
            intent_list = [ind2cls[i] for i in range(len(ind2cls))]
            torch.save(intent_list, intent_list_save_name)
    elif args.label == 'soft':  # from Akshay's data
        with open(args.intent_list, 'rb') as f:
            label_list = pickle.load(f)
            # if args.intent_list == 'data/lr_label_list_no_other.pkl':
            #     correction = {'fix_product': 'fix_product_issue',
            #                   'billing_questions_and_problems': 'billing_question_or_issue'}
            #     label_list = [x if x not in correction else correction[x]
            #                   for x in label_list]
        print('dictionary loaded!')
        cls2ind = {k: i for i, k in enumerate(label_list)}
        ind2cls = {i: k for i, k in enumerate(label_list)}
        intent_list = [ind2cls[i] for i in range(len(ind2cls))]
        torch.save(intent_list, intent_list_save_name)
    else:
        raise NotImplementedError
    print('The intents:')
    print(cls2ind)
    return cls2ind, ind2cls


def initialize_speed_recording_array(batch_size_number):
    array = np.zeros((torch.cuda.device_count() + 1,
                      batch_size_number + 1), dtype=np.float64)
    array[:, 0] = list(range(torch.cuda.device_count() + 1))
    return array


def plot_bar_chart(speed_recording_df, batch_size_list, fig_name):
    speed_recording_df.to_csv(f'{fig_name}.csv')
    ind = np.arange(len(batch_size_list))  # the x locations for the groups
    width = 0.15  # the width of the bars
    labels = ['CPU', '1 GPU', '2 GPUs', '3 GPUs', '4 GPUs']

    fig, ax = plt.subplots()
    for i, row in enumerate(speed_recording_df.values):
        ax.bar(ind - 2 * width + i * width,
               row[1:], width, label=labels[int(row[0])])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('batch size')
    ax.set_ylabel('time per sample (ms)')
    ax.set_title('single sample processing time (lower is better)')
    ax.set_xticks(ind)
    ax.set_xticklabels((str(x) for x in batch_size_list))
    ax.legend()
    fig.savefig(f'{fig_name}.png', dpi=700)
    plt.close()


if __name__ == '__main__':
    os.environ['TORCH_MODEL_ZOO'] = '/home/jovyan/embeddings/deep-sentence-classifiers/tmp/'
    parser = argparse.ArgumentParser(
        description='Train a deep sentence classification model.')
    parser.add_argument('--config', type=str,
                        help='config file path. config will overwrite arguments.')
    parser.add_argument('--memo', type=str,
                        help='Optional. A record of what the job is for.')
    parser.add_argument('--tr', type=str, nargs='+',
                        help='training csv/ json file path')
    parser.add_argument('--db_path', type=str, default='intent.db',
                        help='in what db file shall we store the results and metadata')
    parser.add_argument('--label', type=str, default='hard',
                        help="""use either 'soft' or 'hard' labels""")
    parser.add_argument('--label_column_name', type=str, default='intent',
                        help="""the column name of the classification label, like intent, cit_class, cluster_id""")
    parser.add_argument('--text_column_name', type=str, default='text',
                        help="""the column name of the text input, like customer_message, text""")
    parser.add_argument('--soft_marginals', type=str, required=False,
                        help='if use \'soft\' labels, provide the npy file that contains the soft labels')
    parser.add_argument('--intent_list', type=str, required=False,
                        help='if use soft labels, provide a pickle file of a list of labels, like ["pay_bill", "purchase_product"]')
    parser.add_argument('--va', type=str, nargs='+', required=False,
                        help='validation csv/ json file path')
    parser.add_argument('--save_dir', type=str, default='/data/deep-sentence-classifiers-data/',
                        help='where to save the model config, weights, and predictions')
    parser.add_argument('--encoder', type=str, default='elmo',
                        help='supports bert, elmo, fasttext-godaddy, fasttext-wiki, starspace-godaddy, charcnn, distill-bert.')
    parser.add_argument('--elmo_option_file', type=str, default='',
                        help='when elmo_mode is custom, load this option file.')
    parser.add_argument('--elmo_weights_file', type=str, default='',
                        help='when elmo_mode is custom, load this weights hdf5 file.')
    parser.add_argument('--classifier', type=str, default='att',
                        help='supports maxpooling, gru, att, lp_nlu_att, lp_nlu_att_f_norm, lr, and cnn')
    parser.add_argument('--embedding_type', type=str, default='word',
                        help='supports word, max, mean, cls')
    parser.add_argument('--penalty_C', type=float, default=1,
                        help='lp_nlu_att_f_norm penalty strength')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='supports bert-adam and adam')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (l2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='drop out probability, only work for the cnn classifier')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='mini batch size')
    parser.add_argument('--train_encoder_weights', action='store_true',
                        help='for BERT and ELMo embedders only')
    parser.add_argument('--add_speaker_tag', action='store_true',
                        help='add |||c||| in front of all text')
    parser.add_argument('--datacap', type=int, default=100000000,
                        help='If the dataset size is larger than this, cap the the size of the data file')
    parser.add_argument('--max_sent_len', type=int, default=100,
                        help='cut sentence with more tokens than max_sent_len')
    parser.add_argument('--epoch', type=int, default=300,
                        help='max number of epochs. notice we have early stopping')
    parser.add_argument('--report_interval', type=int, default=20,
                        help='an integer, cf performance after report_interval iterations')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='stop training if in early_stopping many epochs, the performance doesn\'t improve')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the gpu we would like to use.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='for using two gpus at beast server')
    parser.add_argument('--bidirectional_lstm_self_att', action='store_true',
                        help='use bidirectional lstm in the self attention classifier')
    parser.add_argument('--remove_other', action='store_true',
                        help='for removing other labels if the labels are hard')
    parser.add_argument('--fp16', action='store_true',
                        help='half precision. Faster speed but numerically unstable.')
    parser.add_argument('--cpu', action='store_true',
                        help='do not use gpu')
    parser.add_argument('--first_time_inspection', action='store_true',
                        help='do first time inspection')
    parser.add_argument('--cutting_before', action='store_true',
                        help='do first time inspection')
    parser.add_argument('--no_evaluation', action='store_true',
                        help='skip evaluation')
    parser.add_argument('--soft_validation_set', action='store_true',
                        help='skip evaluation')
    parser.add_argument('--va_soft_marginals', type=str, default='data/train_marginals_top_n_merged.npy',
                        help='if use \'soft\' labels, provide the npy file that contains the soft labels')
    parser.add_argument('--override_sanity_check', action='store_true')
    parser.add_argument('--with_tr_report', action='store_true',
                        help='get the training accuracy and performance reported')
    parser.add_argument('--load_model', action='store_true',
                        help='load pretrained model. need to set the hyperparameters the same as the loaded model')
    parser.add_argument('--trained_model_path', type=str, default='',
                        help='the trained model path, like models/2018_12_19_13_01_06_963726.model')
    parser.add_argument('--pretraining', action='store_true',
                        help='two-stage training: first pretraining on some data and then fine-tune')
    parser.add_argument('--pretraining_tr', type=str, default='data/2k_top_n_extended_merged_train.csv',
                        help='training path for the pre-training stage')
    parser.add_argument('--pretraining_va', type=str, default='data/2k_top_n_extended_merged_eval.csv',
                        help='validation path for the pre-training stage')
    parser.add_argument('--pretraining_lr', type=float, default=0.0001,
                        help='learning rate for pretraining')
    parser.add_argument('--pretraining_weight_decay', type=float, default=0.1,
                        help='weight decay for pretraining')
    parser.add_argument('--model_confidence_threshold', type=float, default=0.0,
                        help='a float from 0 to 1. predicts `other` if the predict confidence is less than this')
    parser.add_argument('--data_confidence_threshold', type=float, default=0.0,
                        help='a float from 0 to 1. filter out data with a confidence lower than this one')
    parser.add_argument('--ipc', type=str, default='host')
    parser.add_argument('--shm-size', action='store_true')

    args = parser.parse_args()

    if args.config is not None:
        print(f'loading config file at {args.config}')
        with open(args.config, 'r') as f:
            config_file = yaml.safe_load(f)
        for k, v in config_file.items():
            if k in args.__dict__:
                args.__dict__[k] = v
            else:
                raise KeyError(f'{k} is not a valid argument term')
        print(args)
        
    # initialize path variables, logging, and database, and the GPU/ CPU to use
    logger, model_save_name, va_labels_save_name, intent_list_save_name, timestamp = initialize(
        args)
    db_writer = SQLWriter(db_path=args.db_path)
    db_writer.create_table()
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
    print('DEVICE')
    print(device)
    print(torch.cuda.is_available())

    # generate intent class dict
    cls2ind, ind2cls = generate_intent_list(args, intent_list_save_name)
    cls_num = len(cls2ind)
    intent_list = [ind2cls[i] for i in range(len(ind2cls))]

    # generate the deep intent model
    print(intent_list)
    model = SentenceClassificationModel(intent_list, args)
    if args.fp16:
        model.half()
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # load pretrained model
    if args.load_model:
        logger.info(f'loading model {args.trained_model_path}.')
        model.load_state_dict(torch.load(args.trained_model_path))
        va_data_loader = Loader(args.va, cls2ind, batch_size=args.batch_size,
                                shuffle=False, label='hard', eval=True, args=args, num_total_epochs=5 + 1)
        evaluate(va_data_loader, device, model, 'va', logger,
                 args, mode='plot_single_scores_other_class')
        exit()

    # if need pretraining, do it
    if args.pretraining:
        logger.info('Pretraining starts')
        optimizer = get_optimizer(model, args.pretraining_lr, args.pretraining_weight_decay, args.optimizer)
        train(model, args.pretraining_tr, args.pretraining_va, cls2ind,
              args.batch_size, logger, db_writer, optimizer, device, args)
        logger.info('Pretraining has finished')

    # train the model
    optimizer = get_optimizer(model, args.lr, args.weight_decay, args.optimizer)
    train(model, args.tr, args.va, cls2ind, args.batch_size, logger, db_writer, optimizer, device, args)
