import re
import numpy as np
import json
from os import path
from collections import defaultdict
from math import log
import copy

def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = re.split('\\s+', line)
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

            if character in ['，', '。', '？', '！', '：', '；', '（', '）', '、'] and len(sentence) > 64:
                sentence_list.append(sentence)
                label_list.append(labels)
                sentence = []
                labels = []

    return sentence_list, label_list

def get_word2id(train_data_path):
    word2id = {'<PAD>': 0}
    word = ''
    index = 1
    ############    Had to change how author opened .tsv files to make it work  ###########
    for line in open(train_data_path, 'r', encoding='UTF-8', errors='ignore'):
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            continue
        splits = line.split('\t')
        character = splits[0]
        label = splits[-1][:-1]
        word += character
        if label in ['S', 'E']:
            if word not in word2id:
                word2id[word] = index
                index += 1
            word = ''
    return word2id

################ REWROTE ###############
def get_gram2id(train_data_dir, eval_data_dir) :
    av = {}
    trainLines, _ = read_tsv(train_data_dir)
    testLines, _ = read_tsv(eval_data_dir)
    lines = trainLines + testLines

    # get rid of non interested characters
    newLines = []
    for line in lines :
        newLine = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', ''.join(line))
        for foo in newLine :
            newLines.append(foo)

    ngramDict = {}
    for line in newLines :
        length = len(line)
        for i in range(length) :
            # a word shouldn't be more than 5 Chinese characters
            for j in range(1, 6) :
                if i + j > length :
                    break
                left = i - 1
                right = i + j
                ngram = ''.join(line[left+1:right])
                if ngram in ngramDict :
                    ngramDict[ngram] += 1
                else :
                    ngramDict[ngram] = 1
                    # create sets to allow faster access
                    av[ngram] = {'l': set(), 'r': set()}
                if left >= 0 :
                    av[ngram]['l'].add(line[left])
                if right < length :
                    av[ngram]['r'].add(line[right])

    gram2id = {'<PAD>': 0}
    ngramfreq = {0: 0}
    avfreq = {0: 0}
    ind = 1
    max_len = max([len(x) for x in av.values()])
    max_score = max([min(len(x['l']), len(x['r'])) for x in av.values()])

    for ngram in av.keys() :
        left = len(av[ngram]['l'])
        right = len(av[ngram]['r'])
        score = min(left, right)
        if len(ngram) > max_len :
            max_len = len(ngram)
        if len(ngram) == 1 :
            ngramfreq[ind] = ngramDict[ngram] * 0.267 / 0.698 / max_len
        elif len(ngram) == 2:
            ngramfreq[ind] = ngramDict[ngram] * 0.698 / 0.698 / max_len
        elif len(ngram) == 3:
            ngramfreq[ind] = ngramDict[ngram] * 0.027 / 0.698 / max_len
        elif len(ngram) == 4:
            ngramfreq[ind] = ngramDict[ngram] * 0.00007 / 0.698 / max_len
        else:
            ngramfreq[ind] = ngramDict[ngram] * 0.00002 / 0.698 / max_len

        avfreq[ind] = score / max_score

        ind += 1

    return gram2id, ngramfreq, avfreq

