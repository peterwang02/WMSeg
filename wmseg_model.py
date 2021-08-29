from __future__ import absolute_import, division, print_function

import os

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)
from pytorch_pretrained_bert.tokenization import BertTokenizer

from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.crf import CRF

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'max_ngram_length': 5,
    'do_lower_case': False,
    'do_train': False,
    'use_memory': False
}

############## MODIFIED ##############
class WordKVMN(nn.Module):
    def __init__(self, hidden_size, word_size):
        super(WordKVMN, self).__init__()
        self.temper = hidden_size ** 0.5
        self.word_embedding_a = nn.Embedding(word_size, hidden_size)
        self.word_embedding_c = nn.Embedding(10, hidden_size)
        self.word_embedding_d = nn.Embedding(word_size, hidden_size)
        self.word_embedding_e = nn.Embedding(word_size, hidden_size)

    def forward(self, word_seq, hidden_state, label_value_matrix, word_mask_metrix, freq, avfreq):
        # embedding_a is the embedding for grams (keys), embedding_c is the embedding for labels (values)
        # word_seq is a tensor that has the indices of the ngrams present in the lexicon for each sentence
        # label_value_matrix is a tensor that has (character, ngram) = the label (S, E, B, I)
        # hidden_state is the encoded characters for each sentence
        embedding_a = self.word_embedding_a(word_seq)
        embedding_c = self.word_embedding_c(label_value_matrix)
        embedding_d = self.word_embedding_d(freq)
        embedding_e = self.word_embedding_e(avfreq)
        embedding_a = embedding_a.permute(0, 2, 1)

        # tensor containing every character of each sentence multiplied by every ngram in the sentence
        u = torch.matmul(hidden_state, embedding_a) / self.temper
        # tmp_word_mask_metrix is the label_value_matrix clamped to 0,1
        tmp_word_mask_metrix = torch.clamp(word_mask_metrix, 0, 1)
        exp_u = torch.exp(u)
        # elmentwise operation of [(character, ngram) = value] x [(character, ngram) = label present or not], gives only character-ngram combinations that exist
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)

        # reduce effect of ngrams according to their size and av score

        # delta_exp_u = torch.add(delta_exp_u, avfreq)
        # delta_exp_u = torch.add(delta_exp_u, freq)
        
        # sum up the all the ngram values for each character
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        # p is (character, ngram) = probablity of character forming ngram
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        # (features, ngrams, characters, embeddings) = (embeddings, features, characters, ngrams)
        embedding_c = embedding_c.permute(3, 0, 1, 2)
        embedding_d = embedding_d.permute(3, 0, 1, 2)
        embedding_e = embedding_e.permute(3, 0, 1, 2)

        # probability * label_id
        o = torch.mul(p, embedding_c)
        o = torch.add(o, embedding_d)
        o = torch.add(o, embedding_e)
        # o = (embeddings, features, characters, ngrams)
        
        o = o.permute(1, 2, 3, 0)
        # o = (features, characters, ngrams, embeddings)
        
        # sum up all ngrams for each character
        o = torch.sum(o, 2)
        # o = (features, characters, embeddings)

        o = torch.add(o, hidden_state)

        return o

############## MODIFIED ##############
class WMSeg(nn.Module):

    def __init__(self, word2id, gram2id, gramfreq, av, labelmap, hpara, args):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec.pop('args')
        self.word2id = word2id
        self.gram2id = gram2id
        self.gramfreq = gramfreq
        self.av = av
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.max_ngram_size = self.hpara['max_ngram_size']
        self.max_ngram_length = self.hpara['max_ngram_length']

        self.bert_tokenizer = None
        self.bert = None

        if args.do_train:
            cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
            self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=self.hpara['do_lower_case'])
            print(args.bert_model)
            self.bert = BertModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
            self.hpara['bert_tokenizer'] = self.bert_tokenizer
            self.hpara['config'] = self.bert.config
        else:
            self.bert_tokenizer = self.hpara['bert_tokenizer']
            self.bert = BertModel(self.hpara['config'])
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        if self.hpara['use_memory']:
            self.kv_memory = WordKVMN(hidden_size, len(gram2id))
        else:
            self.kv_memory = None

        self.classifier = nn.Linear(hidden_size, self.num_labels, bias=False)

        self.crf = CRF(tagset_size=self.num_labels - 3, gpu=True)

        if args.do_train:
            self.spec['hpara'] = self.hpara

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, word_seq=None, label_value_matrix=None, word_mask=None, freq=None, avfreq=None):

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if self.kv_memory is not None:
            sequence_output = self.kv_memory(word_seq, sequence_output, label_value_matrix, word_mask, freq, avfreq)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        total_loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
        scores, tag_seq = self.crf._viterbi_decode(logits, attention_mask)

        return total_loss, tag_seq

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['max_ngram_length'] = args.max_ngram_length
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_memory'] = args.use_memory
        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model, args):
        spec = spec.copy()
        res = cls(args=args, **spec)
        res.load_state_dict(model)
        return res

    def load_data(self, data_path, do_predict=False):

        if not do_predict:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
            lines = readfile(data_path, flag=flag)
        else:
            flag = 'predict'
            lines = readsentence(data_path)

        data = []
        for sentence, label in lines:
            if self.kv_memory is not None:
                word_list = []
                matching_position = []
                for i in range(len(sentence)):
                    for j in range(self.max_ngram_length):
                        if i + j > len(sentence):
                            break
                        word = ''.join(sentence[i: i + j + 1])
                        if word in self.gram2id:
                            try:
                                index = word_list.index(word)
                            except ValueError:
                                word_list.append(word)
                                index = len(word_list) - 1
                            word_len = len(word)
                            for k in range(j + 1):
                                if word_len == 1:
                                    l = 'S'
                                elif k == 0:
                                    l = 'B'
                                elif k == j:
                                    l = 'E'
                                else:
                                    l = 'I'
                                matching_position.append((i + k, index, l))
            else:
                word_list = None
                matching_position = None
            data.append((sentence, label, word_list, matching_position))

        examples = []
        for i, (sentence, label, word_list, matching_position) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            if word_list is not None:
                word = ' '.join(word_list)
            else:
                word = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,
                                         label=label, word=word, matrix=matching_position))
        return examples

    ############ MODIFIED ##########
    def convert_examples_to_features(self, examples):

        max_seq_length = min(int(max([len(e.text_a.split(' ')) for e in examples]) * 1.1 + 2), self.max_seq_length)

        if self.kv_memory is not None:
            max_word_size = max(min(max([len(e.word.split(' ')) for e in examples]), self.max_ngram_size), 1)

        features = []

        tokenizer = self.bert_tokenizer

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []
            freq = []
            avfreq = []

            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(self.labelmap[labels[i]])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            if self.kv_memory is not None:
                wordlist = example.word
                wordlist = wordlist.split(' ') if len(wordlist) > 0 else []
                matching_position = example.matrix
                word_ids = []
                matching_matrix = np.zeros((max_seq_length, max_word_size), dtype=np.int)
                freq = np.zeros((max_seq_length, max_word_size), dtype=np.int)
                avfreq = np.zeros((max_seq_length, max_word_size), dtype=np.int)
                if len(wordlist) > max_word_size:
                    wordlist = wordlist[:max_word_size]
                for word in wordlist:
                    try:
                        word_ids.append(self.gram2id[word])
                    except KeyError:
                        print(word)
                        print(wordlist)
                        print(textlist)
                        raise KeyError()
                while len(word_ids) < max_word_size:
                    word_ids.append(0)
                for position in matching_position:
                    char_p = position[0] + 1
                    word_p = position[1]
                    if char_p > max_seq_length - 2 or word_p > max_word_size - 1:
                        continue
                    else:
                        matching_matrix[char_p][word_p] = self.labelmap[position[2]]
                        freq[char_p][word_p] = self.gramfreq[word_p]
                        avfreq[char_p][word_p] = self.av[word_p]


                assert len(word_ids) == max_word_size
            else:
                word_ids = None
                matching_matrix = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              word_ids=word_ids,
                              matching_matrix=matching_matrix,
                              freq=freq,
                              avfreq=avfreq
                              ))
        return features

############## MODIFIED ##############
    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        if self.hpara['use_memory']:
            all_word_ids = torch.tensor([f.word_ids for f in feature], dtype=torch.long)
            all_matching_matrix = torch.tensor([f.matching_matrix for f in feature], dtype=torch.long)
            all_word_mask = torch.tensor([f.matching_matrix for f in feature], dtype=torch.float)
            all_freq = torch.tensor([f.freq for f in feature], dtype=torch.long)
            all_avfreq = torch.tensor([f.avfreq for f in feature], dtype=torch.long)

            word_ids = all_word_ids.to(device)
            matching_matrix = all_matching_matrix.to(device)
            word_mask = all_word_mask.to(device)
            freq = all_freq.to(device)
            avfreq = all_avfreq.to(device)
        else:
            word_ids = None
            matching_matrix = None
            word_mask = None
            avfreq = None
            freq = None

        return input_ids, input_mask, l_mask, label_ids, matching_matrix, segment_ids, valid_ids, word_ids, word_mask, freq, avfreq

############## MODIFIED ##############
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None, matrix=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.matrix = matrix


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, matching_matrix=None, freq=None, avfreq=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.matching_matrix = matching_matrix
        self.freq = freq
        self.avfreq = avfreq


def readfile(filename, flag):
    f = open(filename, encoding='UTF-8')
    data = []
    sentence = []
    label = []

    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            # We randomly concatenate short sentences into long ones if the sentences come from the training set.
            # We do not do that if the sentences come from eval/test set
            if flag == 'train':
                if len(sentence) > 32 or (0 < len(sentence) <= 32 and np.random.rand(1)[0] < 0.25):
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            else:
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
        splits = line.split('\t')
        char = splits[0]
        l = splits[-1][:-1]
        sentence.append(char)
        label.append(l)
        if char in ['，', '。', '？', '！', '：', '；', '（', '）', '、'] and len(sentence) > 64:
            data.append((sentence, label))
            sentence = []
            label = []

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


def readsentence(filename):
    data = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label_list = ['S' for _ in range(len(line))]
            data.append((line, label_list))
    return data

