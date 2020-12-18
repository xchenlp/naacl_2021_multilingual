import time
from math import ceil
import os

import numpy
import torch
import torch.nn.functional as F
from allennlp.modules.elmo import _ElmoCharacterEncoder
from allennlp.modules.elmo import batch_to_ids
from allennlp.modules.elmo import Elmo
from transformers import DistilBertModel, BertModel, RobertaModel, AlbertModel, XLMRobertaModel, DistilBertPreTrainedModel
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from pdb import set_trace
import json

import text_attention

def l2_matrix_norm(m):
    """
    Frobenius norm calculation

    Args:
       m: {Variable} ||AAT - I||

    Returns:
        regularized value


    """
    return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)


def load_model(save_dir, gpu_id):
    """
    this is for loading trained models and make predictions with the model. To use it, do

    ```
    model = load_sentence_classification_model(*args)
    sentences = ['i want to buy an iphone', 'hello']
    predictions, confidence_scores = model.predict(sentences)
    ```

    :param save_dir: Usually the timestamp folder location, such as '/data/deep-sentence-classifiers-data/2019_06_07_13_54_27_060955'.
    :param gpu_id: if use CPU, pass -1
    :return: a trained pytorch model for prediction or embedding
    """
    args_save_name = os.path.join(save_dir, 'args.pt')
    args = torch.load(args_save_name)

    # initialize path variables, logging, and database, and the GPU/ CPU to use
    #device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id != -1) else "cpu")
    device = torch.device("cpu")

    # generate intent class dict
    intent_list_save_name = os.path.join(save_dir, 'intent_list.pt')
    intent_list = torch.load(intent_list_save_name)

    # generate the deep intent model
    #if args.encoder == 'distil-m-bert-fine-tuned':
    #    model = MyDistilBertForClassification('/data/distilmbert_finetuned/', intent_list, args)
    model = SentenceClassificationModel(intent_list, args)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    # load pretrained model
    print(f'loading model {save_dir}')
    model_save_name = os.path.join(save_dir, 'model.pt')
    model.load_state_dict(torch.load(model_save_name))
    model.to(device)

    return model


class LogisticRegression(torch.nn.Module):
    # https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        #sent_emb, _ = torch.max(x, 1)
        #outputs = self.linear(sent_emb)
        outputs = self.linear(x)
        return outputs


class StructuredSelfAttentionLPNLU(torch.nn.Module):
    # from https://github.com/kaushalshetty/Structured-Self-Attention/tree/master/attention
    # modified by Jian Wang

    def __init__(self, hidden_size, cls_num, d_a=350, r=30):
        super(StructuredSelfAttentionLPNLU, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear_first = torch.nn.Linear(hidden_size, d_a, bias=False)
        self.linear_second = torch.nn.Linear(d_a, r, bias=False)
        self.cls_num = cls_num
        self.linear_final = torch.nn.Linear(hidden_size, self.cls_num)
        self.hidden_size = hidden_size
        self.r = r

    def forward(self, x):
        outputs, _ = self.lstm(x)
        x = torch.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = torch.nn.functional.softmax(x, dim=1)
        attention = x.transpose(1, 2)

        # @ for mat multiplication (see https://github.com/pytorch/pytorch/issues/1)
        sentence_embeddings = attention @ outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
        return self.linear_final(avg_sentence_embeddings)

class StructuredSelfAttentionMultilabel(torch.nn.Module):
    # from https://github.com/kaushalshetty/Structured-Self-Attention/tree/master/attention
    # modified by Jian Wang and Kristen Howell for multiclass classification

    def __init__(self, hidden_size, cls_num, d_a=350, r=30):
        super(StructuredSelfAttentionMultilabel, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear_first = torch.nn.Linear(hidden_size, d_a, bias=False)
        self.linear_second = torch.nn.Linear(d_a, r, bias=False)
        self.cls_num = cls_num
        self.linear_final = torch.nn.Linear(hidden_size, self.cls_num)
        self.hidden_size = hidden_size
        self.r = r

    def forward(self, x):
        outputs, _ = self.lstm(x)
        x = torch.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = torch.nn.functional.softmax(x, dim=1)
        attention = x.transpose(1, 2)

        # @ for mat multiplication (see https://github.com/pytorch/pytorch/issues/1)
        sentence_embeddings = attention @ outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
        return self.linear_final(avg_sentence_embeddings)

class StructuredSelfAttentionLPNLUFrobeniusNorm(torch.nn.Module):
    # from https://github.com/kaushalshetty/Structured-Self-Attention/tree/master/attention
    # modified by Jian Wang

    def __init__(self, hidden_size, cls_num, d_a=350, r=30):
        super(StructuredSelfAttentionLPNLUFrobeniusNorm, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear_first = torch.nn.Linear(hidden_size, d_a, bias=False)
        self.linear_second = torch.nn.Linear(d_a, r, bias=False)
        self.cls_num = cls_num
        self.linear_final = torch.nn.Linear(hidden_size, self.cls_num)
        self.hidden_size = hidden_size
        self.r = r

    def forward(self, x):
        outputs, _ = self.lstm(x)
        x = torch.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = torch.nn.functional.softmax(x, dim=1)
        attention = x.transpose(1, 2)

        # @ for mat multiplication (see https://github.com/pytorch/pytorch/issues/1)
        sentence_embeddings = attention @ outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
        return self.linear_final(avg_sentence_embeddings), attention  # also output att for calculating Frobenius norm




class StructuredSelfAttention(torch.nn.Module):
    # from https://github.com/kaushalshetty/Structured-Self-Attention/tree/master/attention
    # modified by Jian Wang

    def __init__(self, hidden_size, cls_num, d_a=350, r=30, bidirectional=False):
        super(StructuredSelfAttention, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            hidden_size *= 2
        self.linear_first = torch.nn.Linear(hidden_size, d_a, bias=False)
        self.linear_second = torch.nn.Linear(d_a, r, bias=False)
        self.cls_num = cls_num
        self.linear_final = torch.nn.Linear(hidden_size, self.cls_num)
        self.hidden_size = hidden_size
        self.r = r
        self.intent_list = None

    def forward(self, sentence_embeddings, output_attention_heatmap_mode=False, encoder_mode=False):
        if output_attention_heatmap_mode and sentence_embeddings.shape[0] != 1:
            raise NotImplementedError('Now only works on batch size 1')
        outputs, _ = self.lstm(sentence_embeddings)

        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        annotation_matrix = torch.nn.functional.softmax(x, dim=1)
        attention = annotation_matrix.transpose(1, 2)
        # @ for mat multiplication (see https://github.com/pytorch/pytorch/issues/1)
        self_attention_sentence_embeddings = attention @ outputs
        avg_self_attention_sentence_embeddings = torch.sum(self_attention_sentence_embeddings, 1) / self.r

        if encoder_mode:
            # output three kinds of embeddings:
            # input sentence_embeddings, lstm word embedding and attention sentence embedding
            return sentence_embeddings, outputs, avg_self_attention_sentence_embeddings

        if output_attention_heatmap_mode:
            return self.linear_final(avg_self_attention_sentence_embeddings), annotation_matrix
        else:

            return self.linear_final(avg_self_attention_sentence_embeddings)

    def output_attention_heatmap(self, sentence_text, sentence_embeddings, true_label, file_name_prefix):
        confidence_score, annotation_matrix = self.forward(sentence_embeddings, output_attention_heatmap_mode=True)

        attention_values = torch.sum(annotation_matrix, 2).cpu().data.numpy()[0]
        _, pred = torch.max(confidence_score, 1)
        pred_label = self.intent_list[pred.cpu().data.numpy()[0]]

        file_name = str(file_name_prefix) + '_' + str(true_label) + '_' + str(pred_label) + '.tex'

        text_attention.generate(sentence_text.split(), attention_values, file_name, 'red')


class CNNClassifier(nn.Module):
    # from git repo https://github.com/galsang/CNN-sentence-classification-pytorch/blob/master/model.py
    def __init__(self, hidden_size, cls_num, args):
        super(CNNClassifier, self).__init__()
        if args.encoder == 'charcnn':
            self.MAX_SENT_LEN = args.max_sent_len + 2
        else:
            self.MAX_SENT_LEN = args.max_sent_len

        self.dropout = args.dropout
        self.WORD_DIM = hidden_size
        self.CLASS_SIZE = cls_num
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = [100, 100, 100]
        self.IN_CHANNEL = 1
        self.embedding_type = args.embedding_type

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            # the f string is so interesting! So is the setattr function
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, embeddings):
        # in eval the last batch may not be of batch size
        embeddings_reshaped = embeddings.view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)

        conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(embeddings_reshaped)), self.MAX_SENT_LEN -
                                     self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in
                        range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)

        return x

# this is modified from BertForMaskedLM in pytorch_pretrained_bert/modeling.py
class SentenceClassificationModel(nn.Module):
    def __init__(self, intent_list, args):
        super(SentenceClassificationModel, self).__init__()
        self.intent_list = intent_list
        self.cls_num = len(intent_list)
        self.classifier_name = args.classifier
        self.encoder_name = args.encoder
        self.label = args.label if 'label' in args.__dict__ else 'hard'
        self.embedding_type = args.embedding_type
        self.args = args
        #config = '/data/distilmbert_finetuned/'
        #self.bert = DistilBertModel.from_pretrained(config)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size, num_labels)
        #self.apply(self.init_bert_weights)

        # encoder
        if self.encoder_name == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'm-bert':
            self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased')#khowell bert team recommends cased over bert-base-multilingual-uncased
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert':
            self.encoder = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'roberta':
            self.encoder = RobertaModel.from_pretrained('roberta-base')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'xlm-roberta':
            self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')#khowell other option is xlm-roberta-base
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'albert':
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == '11-brand-distil-bert':
            self.encoder = DistilBertModel.from_pretrained('/data/11_brand_distill_bert')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == '11-brand-distil-m-bert':
            self.encoder = DistilBertModel.from_pretrained('/data/11_brand_distil_m_bert')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-eng-esp':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_eng_esp')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-esp':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_esp')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-por':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_por')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-eng-esp-por':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_eng_esp_por_new')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-5m':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_5000000/')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-10m':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_10000000/')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-15m':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_15000000/')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-20m':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_20000000/')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-25m':
            self.encoder = DistilBertModel.from_pretrained('/data/distilmbert_25000000/')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-m-bert-fine-tuned':
            self.encoder = DistilBertPreTrainedModel.from_pretrained('/data/distilmbert_finetuned/')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'distil-bert':
            self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.hidden_size = self.encoder.config.hidden_size
            if not args.train_encoder_weights:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif self.encoder_name == 'elmo':
            self.encoder = Elmo(args.elmo_option_file, args.elmo_weights_file, 2,
                                dropout=0, requires_grad=args.train_encoder_weights)

            with open(args.elmo_option_file, "r") as f:
                self.elmo_options = json.load(f)
            
            self.hidden_size = self.elmo_options['lstm']['projection_dim'] * 2

        elif self.encoder_name == 'charcnn':
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

            self.encoder = _ElmoCharacterEncoder(options_file, weight_file,
                                                 requires_grad=True)  # always train encoder weights
            self.hidden_size = 512
        elif self.encoder_name in ('fasttext-godaddy', 'fasttext-wiki', 'starspace-godaddy'):
            self.hidden_size = 300
        else:
            raise NotImplementedError

        # classifier
        if self.classifier_name == 'maxpooling':
            if self.embedding_type == 'first+last':
                self.classifier = MaxpoolingClassifier(self.hidden_size * 2, self.cls_num)
            else:
                self.classifier = MaxpoolingClassifier(self.hidden_size, self.cls_num)
        elif self.classifier_name == 'gru':
            self.classifier = GRUClassifier(self.hidden_size, self.cls_num)
        elif self.classifier_name == 'cnn':
            self.classifier = CNNClassifier(self.hidden_size, self.cls_num, args)
        elif self.classifier_name == 'att':
            bidirectional = self.args.bidirectional_lstm_self_att if 'bidirectional_lstm_self_att' in self.args.__dict__ else False
            print(f'bidirectional LSTM self attention: {bidirectional}')
            self.classifier = StructuredSelfAttention(self.hidden_size, self.cls_num, bidirectional=bidirectional)
        elif self.classifier_name == 'lp_nlu_att':
            self.classifier = StructuredSelfAttentionLPNLU(self.hidden_size, self.cls_num)
        elif self.classifier_name == 'multi_att':
            self.classifier = StructuredSelfAttentionMultilabel(self.hidden_size, self.cls_num)
        elif self.classifier_name == 'lp_nlu_att_f_norm':
            self.classifier = StructuredSelfAttentionLPNLUFrobeniusNorm(self.hidden_size, self.cls_num)
        elif self.classifier_name == 'lr':
            if self.embedding_type == 'first+last':
                self.classifier = LogisticRegression(self.hidden_size*2, self.cls_num)
            else:
                self.classifier = LogisticRegression(self.hidden_size, self.cls_num)
        else:
            raise NotImplementedError

    def _encoder(self, message_tensor, attention_mask):
        if 'bert' in self.encoder_name:
        #if self.encoder_name in ('bert', 'roberta', '11-brand-distil-bert', 'distil-bert', 'albert', 'm-bert', 'xlm-roberta', 'distil-m-bert', '11-brand-distil-m-bert', 'distil-m-bert-eng-esp', 'distil-m-bert-esp', 'dostil-m-bert-esp', 'distil-m-bert-eng-esp-por', 'distil-m-bert-por', 'distil-m-bert-fine-tuned'):
            word_embeddings = self.encoder(message_tensor, attention_mask=attention_mask)[0]
            # Warning: if there are empty messages, BERT will encode them to `nan`, which is undesired
            if self.classifier_name in ['lr', 'maxpooling']:
                if self.embedding_type == 'cls':
                    encoded_layers = word_embeddings[:,0]
                elif self.embedding_type == 'mean':
                    #print(word_embeddings)
                    encoded_layers = torch.mean(word_embeddings, 1)
                    #print(encoded_layers)
                elif self.embedding_type == 'max':
                    encoded_layers, indices= torch.max(word_embeddings, 1)
                    #print(word_embeddings.size())
                    #print(encoded_layers.size())
                elif self.embedding_type == 'first+last':
                    #print('first')
                    #print(word_embeddings[:,0].size())
                    #print('last')
                    #print(word_embeddings[:,-1].size())
                    encoded_layers = torch.cat((word_embeddings[:,0],word_embeddings[:,-1]),0)
                    #print('concatenated')
                    #print(encoded_layers.size())
                else:
                    # khowell: lr and maxpooling both use a single embedding for the sentence. Previously we used the maximum word embedding. This is now handled under the 'max' strategy, so there is no reason to include a 'word' stragy that takes the maximum word embedding.
                    raise NotImplementedError
                #print(word_embeddings.size())
                #print(encoded_layers.size())
                #print(args.max_sent_len)
            elif 'att' in self.classifier_name or self.classifier_name in ['gru','cnn']:
                if self.embedding_type == 'word':
                    encoded_layers = word_embeddings
                else:
                    # khowell: these strategies really only make sense when we look at all of the word embeddings.
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.encoder_name == 'elmo' or self.encoder_name == 'lp-elmo':
            encoded_layers = self.encoder(message_tensor)
            encoded_layers = encoded_layers['elmo_representations'][0] + \
                             encoded_layers['elmo_representations'][1]
        elif self.encoder_name == 'charcnn':
            encoded_layers = self.encoder(message_tensor)
            encoded_layers = encoded_layers['token_embedding']
        elif self.encoder_name in ('fasttext-godaddy', 'fasttext-wiki', 'starspace-godaddy'):
            encoded_layers = message_tensor
        else:
            raise NotImplementedError
        # print('message_tensor')
        # print(torch.sum(message_tensor))
        # print('encoded_layers')
        # print(torch.sum(encoded_layers))
        return encoded_layers

    def forward(self, message_tensor, ground_truth_labels=None, attention_mask=None,
                show_time=False, encoder_mode=False):
        # BERT word embedding
        if message_tensor.shape[1] == 0:
            raise ValueError('The message tensor is empty')
        if show_time:
            start = time.time()

        encoded_layers = self._encoder(message_tensor, attention_mask)
        if show_time:
            end = time.time()
            encode_time = end - start
            #print(f'encoder time: {encode_time}')

        if encoder_mode:
            return self.classifier(encoded_layers, encoder_mode=True)
        # classifier
        if self.classifier_name == 'lp_nlu_att_f_norm':
            confidence_score, att = self.classifier(encoded_layers)
        else:
            confidence_score = self.classifier(encoded_layers)

        # print('confidence_score')
        # sum_conf_score = torch.sum(confidence_score)
        # print(sum_conf_score)
        # if sum_conf_score != sum_conf_score:  # is `nan`
        #     set_trace()
        # print('{k: torch.sum(v) for k, v in self.classifier.state_dict()}')
        # print({k: torch.sum(v) for k, v in self.classifier.state_dict().items()})

        # loss module
        pred = confidence_score >= 0
        #_, pred = torch.max(confidence_score, 1)
        if ground_truth_labels is not None:
            if 'multi' in self.classifier_name:
                loss_fct = BCEWithLogitsLoss()
            elif self.label == 'hard':
                loss_fct = CrossEntropyLoss()
            #elif self.label == 'soft':
            #    # this softmax is not avoidable since the confidence score should sum to 1 (am I right...)
            #    confidence_score = torch.nn.functional.softmax(confidence_score, dim=1)
            #    loss_fct = nn.MSELoss()
            else:
                raise NotImplementedError
            masked_lm_loss = loss_fct(confidence_score, ground_truth_labels)
            if self.label == 'soft':
                masked_lm_loss = 1000.0 * masked_lm_loss  # scale the loss

            if self.classifier_name == 'lp_nlu_att_f_norm':
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1))
                device = next(self.parameters()).device
                identity = identity.to(device)
                identity = identity.unsqueeze(0).expand(att.size(0), att.size(1), att.size(1))
                penalty = self.args.penalty_C * l2_matrix_norm(att @ attT - identity) / att.size(0)
                masked_lm_loss += penalty
            return masked_lm_loss, pred, confidence_score
        else:
            #pred = confidence_score >= 0
            if show_time:
                return pred, confidence_score, encode_time
            return pred, confidence_score

    def _predict_minibatch(self, list_of_tokenized_text, encoder_mode=False):
        device = next(self.parameters()).device
        if self.args.encoder in ('elmo', 'charcnn'):
            attention_mask = None  # don't need it
            message_list = []
            for i, text in enumerate(list_of_tokenized_text):
                tokenized_text = text.split()[:self.args.max_sent_len]
                message_list.append(tokenized_text)
            if self.args.classifier == 'cnn':
                # a hack: padding the message_list
                message_list.append(['n'] * self.args.max_sent_len)
                message_tensor = batch_to_ids(message_list)[:-1, :, :]
            else:
                message_tensor = batch_to_ids(message_list)
        else:
            raise NotImplementedError
        message_tensor = message_tensor.to(device)
        if not encoder_mode:
            pred, confidence_score = self.forward(message_tensor, attention_mask=attention_mask)

            confs = torch.nn.functional.softmax(confidence_score, dim=1).cpu().data.numpy()
            max_confs = numpy.amax(confs, 1)
            return [self.intent_list[x] for x in pred.cpu().data.numpy()], max_confs
        else:
            encoder_word_embeddings, lstm_word_embeddings, attention_sentence_embeddings = self.forward(message_tensor,
                                                                               attention_mask=attention_mask,
                                                                               encoder_mode=True)
            return encoder_word_embeddings.cpu().data.numpy(),\
                   lstm_word_embeddings.cpu().data.numpy(),\
                   attention_sentence_embeddings.cpu().data.numpy()

    def predict(self, list_of_tokenized_text):
        """Follow the input format of sklearn model"""
        self.eval()
        MAX_MINIBATCH_SIZE = 128
        if len(list_of_tokenized_text) <= MAX_MINIBATCH_SIZE:
            return self._predict_minibatch(list_of_tokenized_text)
        else:
            preds = []
            max_confs = []

            minibatch_count = ceil(len(list_of_tokenized_text) / MAX_MINIBATCH_SIZE)
            print(f'{minibatch_count} minibatches in total')
            for i in range(minibatch_count):
                print(f'processing minibatch {i + 1}')
                preds_minibatch, max_confs_minibatch = self._predict_minibatch(
                    list_of_tokenized_text[i * MAX_MINIBATCH_SIZE: (i + 1) * MAX_MINIBATCH_SIZE])
                preds.extend(preds_minibatch)
                max_confs.extend(max_confs_minibatch)
            return preds, max_confs

    def embed(self, list_of_tokenized_text):
        MAX_MINIBATCH_SIZE = 128
        if len(list_of_tokenized_text) <= MAX_MINIBATCH_SIZE:
            return self._predict_minibatch(list_of_tokenized_text, encoder_mode=True)
        else:
            encoder_word_embeddings_all_batchs = []
            lstm_word_embeddings_all_batchs = []
            attention_sentence_embeddings_all_batchs = []

            minibatch_count = ceil(len(list_of_tokenized_text) / MAX_MINIBATCH_SIZE)
            print(f'{minibatch_count} minibatches in total')
            for i in range(minibatch_count):
                print(f'processing minibatch {i + 1}')
                encoder_word_embeddings, lstm_word_embeddings, attention_sentence_embeddings = self._predict_minibatch(
                    list_of_tokenized_text[i * MAX_MINIBATCH_SIZE: (i + 1) * MAX_MINIBATCH_SIZE], encoder_mode=True)
                encoder_word_embeddings_all_batchs.append(encoder_word_embeddings)
                lstm_word_embeddings_all_batchs.append(lstm_word_embeddings)
                attention_sentence_embeddings_all_batchs.append(attention_sentence_embeddings)

            return \
                numpy.concatenate(lstm_word_embeddings_all_batchs, axis=0), \
                   numpy.concatenate(attention_sentence_embeddings_all_batchs, axis=0)

    def output_attention_heatmap(self, sentence_text, message_tensor, true_label, file_name_prefix,
                                 token_type_ids=None, attention_mask=None):
        if not isinstance(self.classifier, StructuredSelfAttention):
            raise NotImplementedError('Only self attention classifier can output attention heatmap')

        self.eval()

        encoded_layers = self._encoder(message_tensor, attention_mask)

        self.classifier.output_attention_heatmap(sentence_text, encoded_layers, true_label, file_name_prefix)


class GRUClassifier(nn.Module):
    def __init__(self, hidden_size, cls_num):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.cls_num = cls_num

        # shall I use tf.nn.rnn_cell.DropoutWrapper for Dropout?
        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.l1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.cls_num)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, encoded_layers):
        _, sent_emb = self.gru(encoded_layers)
        sent_emb = sent_emb[0] + sent_emb[1]  # summarize the two directions

        sent_emb = self.bn1(F.relu(self.l1(sent_emb)))
        sent_emb = self.bn2(F.relu(self.l2(sent_emb)))
        confidence_score = self.l3(sent_emb)
        return confidence_score


class MaxpoolingClassifier(nn.Module):
    def __init__(self, hidden_size, cls_num):
        super(MaxpoolingClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.cls_num = cls_num

        self.l1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.cls_num)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, encoded_layers):
        # maxpooling instead of bert pooling
        # sent_emb, _ = torch.max(encoded_layers, 1)
        sent_emb = encoded_layers
        sent_emb = self.bn1(F.relu(self.l1(sent_emb)))
        sent_emb = self.bn2(F.relu(self.l2(sent_emb)))
        confidence_score = self.l3(sent_emb)
        return confidence_score


