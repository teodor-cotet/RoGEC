import json
from pprint import pprint

import numpy as np
import os
import re
import spacy
import csv
from collections import Counter

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU, Input, concatenate, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, Callback

from keras.backend.tensorflow_backend import set_session

from gensim.models.keyedvectors import KeyedVectors

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from typing import List
from gensim.models.wrappers import FastText as FastTextWrapper

import argparse
args = None


class Model:
    FAST_TEXT = "fasttext_fb/wiki.ro"
    # filename is an elasticsearch dump, use npm elasticdump
    MAX_SENT_TOKENS = 18
    MAX_CHARS_TOKENS = 30
    MAX_ALLOWED_CHAR = 600
    # odd
    WIN_CHARS = 29
    GRU_CELL_SIZE = 64
    PATIENCE = 8
    EPOCHS = 100
    BATCH_SIZE = 64
    DENSES = [128]
    EMB_CHARS_SIZE = 28


    def __init__(self): 
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    # elasticsearch dump file
    def load_data(self, filename='codeDuTravail.json'):
        global args
        id2word, word2id = {}, {}
        count = 0
        text_in, wrong_words, correct_words = [], [], []
        with open(filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for jj, row in enumerate(csv_reader):
                inn = row[1].lower()
                out = row[0].lower()
                in_tokens = inn.split()
                out_tokens = out.split()
                if len(in_tokens) != len(out_tokens):
                    continue
                if args.small_run == True and jj > 5000:
                    continue
                    
                for token in out_tokens:
                    if token not in word2id:
                        word2id[token] = count
                        id2word[count] = token
                        count += 1
                cc = Counter(in_tokens)
                for i, token in enumerate(in_tokens):
                    if token != out_tokens[i] and cc[token] == 1:
                        text_in.append(in_tokens)
                        wrong_words.append(token)
                        correct_words.append(out_tokens[i])
        print(len(text_in))
        
        return text_in, wrong_words, correct_words, id2word, word2id
    

    def split_dataset(self, text_in, wrong_words, correct_words):
        n = len(text_in)
        n1 = int(n * 0.8)

        return text_in[:n1], wrong_words[:n1], correct_words[:n1],\
                text_in[n1:], wrong_words[n1:], correct_words[n1:]

    def construct_window_chars(self, sample, index):
        win_chars = []
        strr = " ".join(sample)
        side = Model.WIN_CHARS // 2

        for i in range(index - side, index + side + 1):
            if i < 0 or i >= len(strr):
                v1 = 0
            elif ord(strr[i]) > Model.MAX_ALLOWED_CHAR:
                v1 = 0
            else:
                v1 = ord(strr[i])
            win_chars.append(v1)
        return win_chars        
    
    def construct_input(self, in_tokens_sent, in_ww):
        if args.small_run == False:
            self.fasttext = FastTextWrapper.load_fasttext_format(Model.FAST_TEXT)
        
        inputs_sent, in_emb_ww, chars_wind = [], [], []
        chars = list(in_tokens_sent)

        for i, sample in enumerate(in_tokens_sent):
            inn = np.zeros((Model.MAX_SENT_TOKENS, 300))
            pos_token = 0
            for j, token in enumerate(sample):
                if args.small_run == True:
                    inn[Model.MAX_SENT_TOKENS - j - 1][:] = np.float32([0] * 300)
                else:
                    try:
                        inn[Model.MAX_SENT_TOKENS - i - 1][:] = np.float32(self.fasttext.wv[token])
                    except:
                        inn[Model.MAX_SENT_TOKENS - j - 1][:] = np.float32([0] * 300)

                if token == in_ww[i]:
                    char_w = self.construct_window_chars(sample, pos_token)
                pos_token += len(token) + 1

            if args.small_run == True:
                try:
                    w_emb = np.float32(self.fasttext.wv[token])
                except:
                    w_emb = np.float32([0] * 300)
            else:
                w_emb = np.float32([0] * 300)

            in_emb_ww.append(w_emb)
            inputs_sent.append(inn)
            chars_wind.append(char_w)

        return [inputs_sent, in_emb_ww, chars_wind]
    
    def construct_output(self, train_cw, word2id):
        out = []
        for i, x in enumerate(train_cw):
            out.append(word2id[x])
        out = keras.utils.to_categorical(out, num_classes=len(word2id))
        return out

    def run_model(self):
        text_in, wrong_words, correct_words, id2word, word2id = self.load_data(filename=args.input_file)
        voc_size = len(id2word)

        train_in, train_ww, train_cw, test_in, test_ww, test_cw = \
            self.split_dataset(text_in, wrong_words, correct_words)

        sentence_embeddings_layer = Input(shape=((Model.MAX_SENT_TOKENS, 300,)))
        sentence_lstm_layer = GRU(units=Model.GRU_CELL_SIZE, input_shape=(Model.MAX_SENT_TOKENS, 300,))
        bi_lstm_layer_sent = keras.layers.Bidirectional(layer=sentence_lstm_layer,\
									merge_mode="concat")(sentence_embeddings_layer)
        word_emb = Input(shape=(300,))
        if args.no_chars == True:
            conc = keras.layers.concatenate([bi_lstm_layer_sent, word_emb], axis=-1)
        else:
            input_character_window = keras.layers.Input(shape=(Model.WIN_CHARS,))
            character_embeddings_layer = keras.layers.Embedding(
                                            input_dim=Model.MAX_ALLOWED_CHAR + 1,\
                                            output_dim=Model.EMB_CHARS_SIZE)(input_character_window)
            chars_lstm_layer = GRU(units=Model.GRU_CELL_SIZE, input_shape=(Model.MAX_CHARS_TOKENS, 300,))
            bi_lstm_layer_chars = keras.layers.Bidirectional(layer=chars_lstm_layer,\
                                        merge_mode="concat")(character_embeddings_layer)
            conc = keras.layers.concatenate([bi_lstm_layer_sent, word_emb, bi_lstm_layer_chars], axis=-1)

        d1 = keras.layers.Dense(Model.DENSES[0], activation='tanh')(conc)                                                 
        output = keras.layers.Dense(voc_size, activation='softmax')(d1)

        train_inn = self.construct_input(train_in, train_ww)
        train_out = self.construct_output(train_cw, word2id)
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=Model.PATIENCE)]
        if args.no_chars == True:
            inn = [sentence_embeddings_layer, word_emb]
        else:
            inn = [sentence_embeddings_layer, word_emb, input_character_window]

        model = keras.models.Model(inputs=inn,
								   outputs=output)
        model.compile(optimizer='adam',\
                    loss='categorical_crossentropy',\
                    metrics=['accuracy'])
        print(model.summary())
        if args.no_chars == True:
            train_inn = [train_inn[0], train_inn[1]]
        model.fit(train_inn,
				  [train_out],
				  batch_size=Model.BATCH_SIZE, 
                  epochs=Model.EPOCHS, 
                  validation_split=0.2,
                  callbacks=callbacks)
        model.save(args.name + '.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--small_run', dest='small_run', action='store_true', default=False)
    parser.add_argument('--name', dest="name", action="store", default="default")
    parser.add_argument('--no_chars', dest="no_chars", action="store_true")
    parser.add_argument('--input_file', dest="input_file", action="store", default="inflected.csv")
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])

    model = Model()
    model.run_model()
    
