import random

import os
import torch
from pdb import set_trace
import numpy as np
import pandas as pd
import torch
from allennlp.modules.elmo import batch_to_ids
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer, XLMRobertaTokenizer
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
import time
random.seed(0)


class Loader(object):
    def __init__(self, data_file_name, cls2ind, num_total_epochs, batch_size, args, shuffle=True,
                 datacap=None, eval=False, label='hard', soft_marginals=None, first_time_inspection=False):
        """
        Takes in tokenized csv data and convert them to PyTorch tensors.
        :param data_file_name: file containing the text and (in label=='hard' case) ground truth class labels
        :param cls2ind: a dictionary that maps label classes to indices
        :param batch_size: mini batch size
        :param shuffle: whether to shuffle data after each epoch
        :param datacap: how much data to use. If the size is larger than the data file size, will ignore this number
        :param eval: evaluation mode. turn it on when doing validation or testing
        :param label: 'soft' or 'hard' or 'none'
        :param soft_marginals: the name of the npy file containing soft ground truth labels
        :param first_time_inspection: observe the soft marginals when first time running
        """

        # record the hyperparameters for the loader
        self.args = args
        # If all sentence in a mini-batch is shorter than self.args.max_sent_len,
        # whether to pad to self.args.max_sent_len
        # cnn classifier doesn't support variable length so set this to True when using cnn classifier
        if 'cutting_before' in self.args.__dict__ and self.args.cutting_before:
            print('cutting message before instead of after')
        self.pad_to_max_sent_len = True if args.classifier == 'cnn' else False
        self.data_file_name = data_file_name
        self.batch_size = batch_size
        self.add_speaker_tag = args.add_speaker_tag if 'add_speaker_tag' in args.__dict__ else False

        self.shuffle = shuffle
        self.datacap = datacap
        self.eval = eval
        self.label = label
        self.num_total_epochs = num_total_epochs
        self.cls2ind = cls2ind
        self.ind2cls = {v: k for k, v in cls2ind.items()}
        self.class_num = len(cls2ind)
        self.soft_marginals = soft_marginals

        self._get_tokenizer()
        if self.args.encoder in ('fasttext-godaddy', 'fasttext-wiki', 'starspace-godaddy'):
            self._load_word2vec_dict(self.args.encoder)

        # set counters
        self.epoch = 1
        self.batch_idx = -1  # in current epoch
        self.total_iter = -1  # total iteration

        self._load_data()
        if self.shuffle:
            random.shuffle(self.data)

        # observe the data
        print('Reminder: observe the data when first time running it!')
        if first_time_inspection:
            self._first_time_inspection()

        # keep the data size we want
        if datacap is not None:
            self.data = self.data[:datacap]

        # measure parameters of the data
        self.data_size = len(self.data)
        self.total_batch = self.data_size // self.batch_size
        # in evaluation mode, we should evaluate the last chunk of data even though the size < batch_size
        if self.eval and (len(self.data) % self.batch_size) != 0:
            self.total_batch += 1

        # create a queue for each pytorch tensor
        self.queue_maxsize = 5
        self.message_tensor_queue = Queue(self.queue_maxsize)
        self.label_tensor_queue = Queue(self.queue_maxsize)
        self.attention_mask_queue = Queue(self.queue_maxsize)

        # start the data loading process
        self.reader = Process(target=self.read)
        self.reader.daemon = True
        self.reader.start()
        # self.reader.join()

    def _first_time_inspection(self):
        if self.label == 'soft':
            for i in range(20):
                text, confidence = random.choice(self.data)
                print('text: ', text)
                print('label: ', self.ind2cls[int(np.argmax(confidence))])
                print('confidence: ', float(np.max(confidence)))
                print('confidence marginals: ', confidence, '\n')
            print(
                f'the most confident confidence score is {np.max(self.soft_labels)}')
        elif self.label == 'hard':
            print('hard label inspections to be implemented')
        exit()

    def _load_data(self):
        df = pd.DataFrame()
        for file in self.data_file_name:
            print('FILE NAME')
            print(file)
            if file.endswith('csv'):
                df = df.append(pd.read_csv(file),sort=False)
            elif file.endswith('json'):
                df = df.append(pd.read_json(file, lines=True),sort=False)
            else:
                raise NotImplementedError

        if self.label == 'soft':
            # read via pandas and np.load
            self.soft_labels = np.load(self.soft_marginals)

            # load the data to a list
            texts_up_to_last_label = df[self.args.text_column_name].iloc[:self.soft_labels.shape[0]]
            # binary classification case. pad to shape (length, 2)
            if len(self.soft_labels.shape) == 1:
                temp = np.ones((self.soft_labels.shape[0], 2), dtype=float)
                temp[:, 1] = self.soft_labels
                temp[:, 0] -= self.soft_labels
                self.soft_labels = temp
            self.data = [(text, self.soft_labels[i, :].astype(np.float32))
                         for i, text in enumerate(texts_up_to_last_label)]

        elif self.label == 'hard':
            # preprocessing
            if self.args.data_confidence_threshold > 0.0:
                print(f'cutting the low-confidence data. Original data size {len(df)}')
                df = df.loc[df.fig8_confidence >
                            self.args.data_confidence_threshold, :]
                print(f'now the data size is {len(df)}')
            if self.args.remove_other:
                df = df[df[self.args.label_column_name] != 'other']
                df = df[df[self.args.label_column_name] != 'Other']
          
            # load the data to a list
            if 'multi' in self.args.classifier:
                # if multilabel classification, format labels as a binary list
                df[self.args.label_column_name] = df[self.args.label_column_name].map(lambda label: label.split('|'))
                df[self.args.label_column_name] = df[self.args.label_column_name].map(lambda x: self._indices2binarylist(x))
                self.data = [(eval(f'row.{self.args.text_column_name}'),
                              eval(f'row.{self.args.label_column_name}'))
                              for row in df.itertuples() if isinstance(eval(f'row.{self.args.text_column_name}'), str)]
            else:
                self.data = [(eval(f'row.{self.args.text_column_name}'),
                          self.cls2ind[eval(f'row.{self.args.label_column_name}')])
                         for row in df.itertuples() if isinstance(eval(f'row.{self.args.text_column_name}'), str)]
        elif self.label == 'none':
            # load the data to a list
            self.data = [(eval(f'row.{self.args.text_column_name}'), None) for row in df.itertuples() if isinstance(eval(f'row.{self.args.text_column_name}'), str)]

        self.df = df

    def _indices2binarylist(self, indices):
        binary_list = [0 for x in range(0, self.class_num)]
        for i in indices:
            binary_list[int(self.cls2ind[i])] = 1
        return(np.array(binary_list, dtype=np.int64))

    def _load_word2vec_dict(self, encoder):
        self.embedding_size = 300
        print(f'loading {encoder} embeddings')
        self.word2vec = {}
        # load the word2vec dict
        if encoder == 'fasttext-godaddy':
            model_path = 'model_weights/godaddy_model_300.vec'
        elif encoder == 'fasttext-wiki':
            model_path = '/data/cc.en.300.vec'
            model_torch_pickled_path = '/data/cc.en.300.vec.pt'
            if os.path.isfile(model_torch_pickled_path):
                print('loading torch pickled file to save time')
                self.word2vec = torch.load(model_torch_pickled_path)
                return
        elif encoder == 'starspace-godaddy':
            model_path = 'model_weights/godaddy_jian.starspace.tsv'
        else:
            raise NotImplementedError

        start = time.time()
        with open(model_path, 'rt') as f:
            for line in f:
                vf = []
                if encoder in ('fasttext-godaddy', 'fasttext-wiki'):
                    word, *vec = line.rstrip().split()
                elif encoder == 'starspace-godaddy':
                    word, *vec = line.rstrip().split('\t')
                else:
                    raise NotImplementedError
                for v in vec:
                    vf.append(float(v))
                if len(vf) == self.embedding_size:
                    self.word2vec[word] = np.array(vf, dtype=np.float32)

                else:
                    print(f'found word {word} not conform to the embedding length. the embedding is {vf}')
                    print('this should only filter out the first line of the vec file for fasttext')
                    print(f'the line is {line}')
        end = time.time()
        print(f'finished loading, {len(self.word2vec)} words loaded.')
        print(f'for example, the embedding of `you` is {self.word2vec["you"]}')
        print(f'loading time is {end-start}')
        if encoder == 'fasttext-wiki':
            print('save torch pickled file to save time')
            torch.save(self.word2vec, model_torch_pickled_path)

    def _get_tokenizer(self):
        """
        initialize the tokenizer
        """
        if self.args.encoder == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',truncation=True)
        elif self.args.encoder == 'm-bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',truncation=True)#khowell bert team recommends cased over bert-base-multilingual-uncased
        elif 'distil-m-bert' in self.args.encoder:
        #elif self.args.encoder in ['distil-m-bert', '11-brand-distil-m-bert', 'distil-m-bert-eng-esp', 'distil-m-bert-eng-esp-por', 'distil-m-bert-por', 'distil-m-bert-esp', 'distil-m-bert-fine-tuned']:
            self.tokenizer = DistilBertTokenizer.from_pretrained('bert-base-multilingual-cased',truncation=True)
        elif self.args.encoder in ['distil-bert', '11-brand-distil-bert']:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',truncation=True)
        elif self.args.encoder == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base',truncation=True)
        elif self.args.encoder == 'xlm-roberta':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base',truncation=True)#khowell other option is xlm-roberta-large
        elif self.args.encoder == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2',truncation=True)
        # elmo and charcnn share the same input
        elif self.args.encoder in ('elmo', 'charcnn', 'lp-elmo', 'fasttext-godaddy', 'fasttext-wiki', 'starspace-godaddy'):
            self.tokenizer = lambda x: x.split()

    def next_batch(self):
        """
        get the next mini batch of data from the queues
        :return: a tuple of (message_tensor, label_tensor, attention_mask)
        """
        data = (self.message_tensor_queue.get(),
                self.label_tensor_queue.get(), self.attention_mask_queue.get())
        if data[0] is None:
            data = (self.message_tensor_queue.get(),
                    self.label_tensor_queue.get(), self.attention_mask_queue.get())
            self.epoch += 1
            self.batch_idx = 0
            self.total_iter += 1
        else:
            self.batch_idx += 1
            self.total_iter += 1
        return data

    def read(self):
        """
        a process that forever runs: call self.prepare function and put the prepared mini batches to the queues
        """
        for _ in range(self.num_total_epochs):  # each iter of the while loop is an epoch
            if self.shuffle:
                random.shuffle(self.data)

            # prepare data
            for i in range(self.total_batch):
                minibatch_message_tensor, minibatch_label_tensor, minibatch_attention_mask = self.prepare(self.data[i * self.batch_size: (i + 1) * self.batch_size])
                self.message_tensor_queue.put(minibatch_message_tensor)
                self.label_tensor_queue.put(minibatch_label_tensor)
                self.attention_mask_queue.put(minibatch_attention_mask)

            # indicate that an epoch has finished
            self.message_tensor_queue.put(None)
            self.label_tensor_queue.put(None)
            self.attention_mask_queue.put(None)

    def prepare(self, minibatch_data):
        """
        Prepare a minibatch of data.
        :param minibatch_data: mini batch data
        :return: message_tensor, label_tensor, attention_mask (three torch.tensor objects)
        """
        minibatch_size = len(minibatch_data)  # for the last batch in eval mode, the minibatch_size may not be the batch size

        # prepare the ground truth labels
        if self.label == 'hard':
            label_tensor = np.array([np.array(label) for _, label in minibatch_data], dtype=np.float32)
            label_tensor = torch.tensor(label_tensor)
        elif self.label == 'soft':
            label_tensor = np.stack([label for _, label in minibatch_data])
            assert label_tensor.dtype == np.float32, \
                f'wrong data format: not np.float32, instead it is {label_tensor.dtype}'
            label_tensor = torch.tensor(label_tensor)
        elif self.label == 'none':
            label_tensor = None
        else:
            raise NotImplementedError

        # prepare the message_tensor and attention_mask
        if 'bert' in self.args.encoder:
        #if self.args.encoder in ('bert', 'roberta', 'albert', 'distil-bert', '11-brand-distil-bert', 'm-bert', 'xlm-roberta', 'distil-m-bert', '11-brand-distil-m-bert', 'distil-m-bert-eng-esp', 'distil-m-bert-eng-esp-por', 'distil-m-bert-por', 'distil-m-bert-esp', 'distil-m-bert-fine-tuned'):
            message_tensor = np.zeros((minibatch_size, self.args.max_sent_len), dtype=np.int64)
            attention_mask = np.zeros((minibatch_size, self.args.max_sent_len), dtype=np.int64)
            for i, example in enumerate(minibatch_data):
                text, label = example
                if self.add_speaker_tag:
                    text = '|||c||| ' + text
                if 'cutting_before' in self.args.__dict__ and self.args.cutting_before:
                    pass  # not implemented
                # tokenize and cut
                message = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.args.max_sent_len,truncation=True)
                msg_len = len(message)
                message_tensor[i, :msg_len] = message[:msg_len]
                attention_mask[i, :msg_len] = 1
            message_tensor = torch.tensor(message_tensor)
            attention_mask = torch.tensor(attention_mask)
        # elmo and charcnn share the same input
        elif self.args.encoder in ('elmo', 'charcnn'):
            attention_mask = None  # don't need it
            message_list = []
            for i, example in enumerate(minibatch_data):
                text, label = example
                if self.add_speaker_tag:
                    text = '|||c||| ' + text
                if 'cutting_before' in self.args.__dict__ and self.args.cutting_before:
                    tokenized_text = self.tokenizer(text)[-self.args.max_sent_len:]
                else:
                    tokenized_text = self.tokenizer(text)[:self.args.max_sent_len]
                message_list.append(tokenized_text)
            if self.pad_to_max_sent_len:
                # a hack: padding the message_list
                message_list.append(['n'] * self.args.max_sent_len)
                message_tensor = batch_to_ids(message_list)[:-1, :, :]
            else:
                message_tensor = batch_to_ids(message_list)
        elif self.args.encoder in ('fasttext-godaddy', 'fasttext-wiki', 'starspace-godaddy'):
            attention_mask = None  # don't need it
            message_tensor = np.zeros((minibatch_size, self.args.max_sent_len,
                                       self.embedding_size), dtype=np.float32)  # this is just the embedding
            for i, example in enumerate(minibatch_data):
                text, label = example
                if self.add_speaker_tag:
                    text = '|||c||| ' + text
                if 'cutting_before' in self.args.__dict__ and self.args.cutting_before:
                    tokenized_text = self.tokenizer(text)[-self.args.max_sent_len:]
                else:
                    tokenized_text = self.tokenizer(text)[:self.args.max_sent_len]

                for j, token in enumerate(tokenized_text):
                    if token in self.word2vec:
                        weight = self.word2vec[token]
                    else:
                        # https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
                        # it is discussable how to process the <UNKNOWN> embedding
                        weight = np.random.normal(scale=0.6, size=(self.embedding_size,))
                    message_tensor[i, j, :] = weight
            message_tensor = torch.tensor(message_tensor)
        else:
            raise NotImplementedError

        return message_tensor, label_tensor, attention_mask

    def destruct(self):
        self.reader.terminate()
