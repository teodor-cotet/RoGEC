# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from pprint import pprint
from typing import List, Tuple

import numpy as np
import spacy
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText as FastTextWrapper
from keras.backend.tensorflow_backend import set_session
from sklearn import datasets, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from tensorflow import keras
from tensorflow.keras.callbacks import (Callback, LambdaCallback,
                                        ModelCheckpoint)
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout,
                                     Embedding, Input, RepeatVector, Reshape,
                                     TimeDistributed, concatenate)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from parser import Parser

args = None
""" TODO:
    1. keep chars based on frequences, not UNICODE codepoint
    2. batches with different sequences lengths
"""

log = open("log.log", "w", encoding='utf-8')

class Model:
    FAST_TEXT = "fasttext_fb/wiki.ro"

    """ params of the model """
    PATIENCE = 1000
    EPOCHS = 200
    BATCH_SIZE = 10
    LATENT_DIM_RNN = 32
    LATENT_DIM_CHARS = 16
    MAX_LENGTH_DECODER = 110

    """ max char to keep """
    MAX_CHAR = 150
    START_CHAR = 150
    END_CHAR = 150 + 1
    NR_CHARS = 150 + 2

    SRC_TEXT_CHAR_LENGTH = 200
    SMALL_RUN_SAMPLES = 400
    TRAIN_DEV_DATASET_PERCENTAGE = 0.97

    CORRECT_DIACS = {
        "ş": "ș",
        "Ş": "Ș",
        "ţ": "ț",
        "Ţ": "Ț",
    }

    def __init__(self): 
        pass
        # config = tf.ConfigProto()
        # #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        # config.gpu_options.allow_growth = True
        # set_session(tf.Session(config=config))

    # elasticsearch dump file
    def load_data(self, filename):
        global args
        id2word, word2id = {}, {}
        count = 0
        raw_in, raw_out, text_in, text_out, wrong_words, correct_words = [], [], [], [], [], []

        with open(filename, "r", encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for jj, row in enumerate(csv_reader):
                inn = self.clean_text(row[1].lower())
                out = self.clean_text(row[0].lower())
                throw = False
                """ discard long texts - TODO we want to keep those """
                if len(inn) > Model.SRC_TEXT_CHAR_LENGTH - 2 or len(out) > Model.SRC_TEXT_CHAR_LENGTH - 2:
                    throw = True

                """ discard sent with weird characters ord(c) > MAX_CHAR """
                for c in list(inn):
                    if ord(c) == 0 or ord(c) >= Model.MAX_CHAR:
                        throw = True

                for c in list(out):
                    if ord(c) == 0 or ord(c) >= Model.MAX_CHAR:
                        throw = True
                
                in_tokens = inn.split()
                out_tokens = out.split()

                """ discard sent with diff nr of tokens """
                if len(in_tokens) != len(out_tokens):
                    continue

                if args.small_run == True and jj > Model.SMALL_RUN_SAMPLES:
                    continue
                if jj > 300e3:
                    continue
                if throw == True:
                    continue

                for token in out_tokens:
                    if token not in word2id:
                        word2id[token] = count
                        id2word[count] = token
                        count += 1
                """ dicard when problematic token repeats """
                cc = Counter(in_tokens)
                for i, token in enumerate(in_tokens):
                    # keep only if token does not repeat
                    if token != out_tokens[i] and cc[token] == 1:
                        raw_in.append(inn)
                        raw_out.append(out)
                        text_in.append(in_tokens)
                        text_out.append(out_tokens)
                        wrong_words.append(token)
                        correct_words.append(out_tokens[i])
        return raw_in, raw_out, text_in, text_out, wrong_words, correct_words, id2word, word2id
    
    def clean_text(self, text: str):
        list_text = list(text)
        text = "".join([Model.CORRECT_DIACS[c] if c in Model.CORRECT_DIACS else c for c in list_text])
        return text.lower()

    def split_dataset(self, raw_in, raw_out, text_in, text_out, wrong_words, correct_words):
        n = len(raw_in)
        n1 = int(n * Model.TRAIN_DEV_DATASET_PERCENTAGE)

        return raw_in[:n1], raw_out[:n1], text_in[:n1], text_out[:n1], wrong_words[:n1], correct_words[:n1],\
                raw_in[n1:], raw_out[n1:], text_in[n1:], text_out[n1:], wrong_words[n1:], correct_words[n1:]      
    
    def pad_seq(self, seq, max_size):
        seq.insert(0, Model.START_CHAR)
        """no need to truncate, all have len < Model.SRC_TEXT_CHAR_LENGTH - 2"""
        padded_seqs = pad_sequences(sequences=[seq], maxlen=(max_size - 1), padding='pre', value=0)[0]
        final_padded_seqs = np.append(padded_seqs, Model.END_CHAR)
        return np.asarray(final_padded_seqs)

    def construct_input_output_chars(self, raw_in, raw_out, max_size):
        global args
        """ all_samples_decoder_target us one timestep ahead of all_samples_decoder_input (teacher forcing)
            it is also categorical """
        all_samples_decoder_target, all_samples_encoder, all_samples_decoder_input = [], [], []
        for (rin, rout) in zip(raw_in, raw_out):
            """ in """
            chars_ids = [ord(c) for c in list(rin)]
            padded_in = self.pad_seq(chars_ids, max_size=max_size)
            all_samples_encoder.append(padded_in)      
            """ out """
            chars_ids = [ord(c) for c in list(rout)]
            padded_out = self.pad_seq(chars_ids, max_size=max_size)
            all_samples_decoder_input.append(padded_out)
            skipped_first_start_char, sent_samples_out = False, []

            for out in padded_out:
                """ skip the first character, because target decoder has to be 1 timestep ahead """
                if out == Model.START_CHAR and skipped_first_start_char == False:
                    skipped_first_start_char = True
                else:
                    sent_samples_out.append(out)
            """ append 0 to match sizes"""
            sent_samples_out.append(0)
            assert len(sent_samples_out) == max_size, 'decoder target length mismatch'
            all_samples_decoder_target.append(np.asarray(sent_samples_out))

        return np.asarray(all_samples_encoder), np.asarray(all_samples_decoder_input), np.asarray(all_samples_decoder_target)
    
    """"returns list with (probability, index) """
    def get_k_most_probable(self, probs, k) -> List:
        probs_indices = [i for i, _ in enumerate(probs)]
        probs_indices = sorted(probs_indices, key = lambda x: probs[x], reverse=True)
        return [(probs[i], i) for i in probs_indices[:k]]
    
    def update_candidates(self, k_potential_cands, new_beam_cands, base_str, state, base_prob):

        for cand in k_potential_cands:
            """ find lowest probability from the new beam candidates """
            lowest_prob_key = None
            for key_cand, new_cand in new_beam_cands.items():
                new_prob, _ = new_cand
                if lowest_prob_key is None or new_prob < new_beam_cands[lowest_prob_key][0]:
                    lowest_prob_key = key_cand
            
            (cand_prob, index) = cand
            new_str = base_str + chr(index)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = index
            saved_target_and_state = [target_seq] + state

            if len(new_beam_cands) < args.beam_width:
                new_beam_cands[new_str] = (cand_prob * base_prob, saved_target_and_state) 
            else:
                if cand_prob * base_prob > new_beam_cands[lowest_prob_key][0]:
                    del new_beam_cands[lowest_prob_key]
                    
                    # Populate the first character of target sequence with the start character.
                    
                    new_beam_cands[new_str] = (cand_prob * base_prob, saved_target_and_state) 

    def compute_predictions(self, input_seq):
        """decoding works as follows:
            # Here's the drill:
            # 1) encode input and retrieve initial decoder state
            # 2) run one step of decoder with this initial state
            #   and a "start of sequence" token as target (first input)
            #   Output will be the next target token
            # 3) Repeat with the current target token and current states. You predict 1 step at a time
                 , changes states of the lstm (cell and hidden) by yourself.
                 Decoder model is constructed such that it returns not only the output, but the cell and hidden state,
                 so that you can manipulate them.
        """
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = Model.START_CHAR

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence, decoded_ids = '', []
        """ beam_cand[str] = (prob, [target_seq] + states_value) """
        beam_candidates = {chr(Model.START_CHAR): (1.0, [target_seq] + states_value)}

        for i in range(Model.MAX_LENGTH_DECODER):
            new_beam_candidates = {}
            for base_str, candidate in beam_candidates.items():
                (base_prob, base_state) = candidate
                output_tokens, h, c = self.decoder_model.predict(base_state)
                states_value = [h, c]
                k_candidates = self.get_k_most_probable(output_tokens[0, -1, :], k=args.beam_width)
                self.update_candidates(k_candidates, new_beam_cands=new_beam_candidates, 
                                        base_str=base_str, state=states_value, base_prob=base_prob)

                """Exit condition: either hit max length, or find stop character TODO"""
            beam_candidates = new_beam_candidates
        for decoded_str, v in beam_candidates.items():
                print('decoded seq: {}, with prob: {}'.format(decoded_str, v[0]), file=log)
        return beam_candidates

    def run_model_char_decoder(self):
        global args

        raw_in, raw_out, text_in, text_out, wrong_words, correct_words, id2word, word2id = self.load_data(filename=args.input_file)
        lengths = Counter([len(txt_in) for txt_in in text_in])
        print('text lengths: {}'.format(lengths))

        train_raw_in, train_raw_out, train_in, train_out, train_ww, train_cw,\
             test_raw_in, test_raw_out, test_in, test_out, test_ww, test_cw = \
                self.split_dataset(raw_in, raw_out, text_in, text_out, wrong_words, correct_words)
        
        # teacher forcing method used for trainings
        if args.no_train == False:
            print('Train...', file=log)
            vocab_size = Model.NR_CHARS
            
            # encoder input model
            encoder_inputs = Input(shape=(None,))
            char_emb_encoder = Embedding(input_dim=Model.NR_CHARS, output_dim=Model.LATENT_DIM_CHARS, mask_zero=True, trainable=True)
            embedded_encoder = char_emb_encoder(encoder_inputs)
            # states used later
            encoder_lstm, encoder_h, encoder_c = LSTM(Model.LATENT_DIM_RNN, return_state=True)(embedded_encoder)
            # hidden and cell state for the last timestep

            encoder_states = [encoder_h, encoder_c]
            # decoder output model, we define a new Input because is recursive
            decoder_inputs = Input(shape=(None,))
            char_emb_decoder = Embedding(input_dim=Model.NR_CHARS, output_dim=Model.LATENT_DIM_CHARS, mask_zero=True, trainable=True)
            embedded_decoder = char_emb_decoder(decoder_inputs)
            # ret seq return output for each timestep, retur_state return cell and hidden 
            decoder_lstm = LSTM(Model.LATENT_DIM_RNN, return_sequences=True, return_state=True)
            decoder_outputs, _, _ = decoder_lstm(embedded_decoder, initial_state=encoder_states)
            decoder_dense = Dense(Model.NR_CHARS, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)

            model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

            model.compile(loss='categorical_crossentropy', optimizer='adam')

            train_samples_in, train_out_simple, train_out_categorical = \
                self.construct_input_output_chars(raw_in=train_raw_in, raw_out=train_raw_out, 
                                                  max_size=Model.SRC_TEXT_CHAR_LENGTH)
            train_samples_in.reshape(())
            if args.verbose:
                print('Coding in/out info')
                for (inn, outs, outc) in zip(train_samples_in, train_out_simple, train_out_categorical):
                    print('input seq: ')
                    print(inn, file=log)
                    print('input shape: ')
                    print(inn.shape, file=log)
                    print('input (decoder) seq:')
                    print(outs, file=log)
                    print('input (decoder) shape:')
                    print(outs.shape, file=log)
                    print('output (train) seq:')
                    print(outc, file=log)
                    print('output (train) shape:')
                    print(outc.shape, file=log)
                    break

            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=Model.PATIENCE)]
            print(model.summary())
            model.fit([train_samples_in, train_out_simple],
                    [train_out_categorical],
                    batch_size=Model.BATCH_SIZE, 
                    epochs=Model.EPOCHS, 
                    validation_split=0.2,
                    callbacks=callbacks,
                    shuffle='batch')
            
            """ construct prediction model
                we need to construct a new one because we used teacher forcing (where the input for a timestep is not 
                not the previous generated output, but the correct output) """
            self.encoder_model = keras.Model(encoder_inputs, encoder_states)

            decoder_state_input_h = Input(shape=(Model.LATENT_DIM_RNN,))
            decoder_state_input_c = Input(shape=(Model.LATENT_DIM_RNN,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs, state_h, state_c = decoder_lstm(
                embedded_decoder, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model = keras.Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

            test_samples_in, test_out_simple, test_out_categorical =\
                 self.construct_input_output_chars(raw_in=test_raw_in, raw_out=test_raw_out, 
                                                    max_size=Model.SRC_TEXT_CHAR_LENGTH)
            
            for input_seq in train_samples_in[:20]:
                print(input_seq, file=log)
                self.compute_predictions(input_seq)
        else:
           model = keras.models.load_model(args.load)

    def filter_texts(self, texts):
        """ filter texts with weird characters and high length """
        filtered = []
        for t1, t2 in texts:
            keep = True
            for c in t1:
                if ord(c) >= Model.NR_CHARS:
                    keep = False
            for c in t2:
                if ord(c) >= Model.NR_CHARS:
                    keep = False
            if len(t1) >= Model.SRC_TEXT_CHAR_LENGTH - 2 or len(t2) > Model.SRC_TEXT_CHAR_LENGTH - 2:
                keep = False
            if keep:
                filtered.append((t1, t2))
        return filtered

    def eng_noiser(self):
        parser = Parser()
        train, test = parser.parse_lang_8()
        train, test = self.filter_texts(train), self.filter_texts(test)

        if args.small_run:
            train, test = train[:Model.SMALL_RUN_SAMPLES], test[:Model.SMALL_RUN_SAMPLES]

        train_raw_in, train_raw_out = [t1 for t1, _ in train], [t2 for _, t2 in train]
        #train_raw_out = train_raw_in
        test_raw_in, test_raw_out = [t1 for t1, _ in test], [t2 for _, t2 in test]
        # model seq2seq
        # encoder input model
        encoder_inputs = Input(shape=(None,))
        char_emb_encoder = Embedding(input_dim=Model.NR_CHARS, output_dim=Model.LATENT_DIM_CHARS, mask_zero=True, trainable=True)
        embedded_encoder = char_emb_encoder(encoder_inputs)
        """ LSTM has cell state and hidden (state) == output 
            states used later, encoder_h == encoder_lstm  (last hidden states == output)
            for full seq, call return_sequences==True
        """
        encoder_lstm, encoder_h, encoder_c = LSTM(Model.LATENT_DIM_RNN, return_state=True)(embedded_encoder)

        encoder_states = [encoder_h, encoder_c]
        # decoder output model, we define a new Input because is recursive
        decoder_inputs = Input(shape=(None,))
        char_emb_decoder = Embedding(input_dim=Model.NR_CHARS, output_dim=Model.LATENT_DIM_CHARS, mask_zero=True, trainable=True)
        embedded_decoder = char_emb_decoder(decoder_inputs)
        """ ret seququences return output (hidden) for each timestep, retur_state returns cell and hidden """
        decoder_lstm = LSTM(Model.LATENT_DIM_RNN, return_sequences=True, return_state=True)
        """ initial_state is the initial cell and initial hidden, which are initialized with the encoder output """
        decoder_outputs, _, _ = decoder_lstm(embedded_decoder, initial_state=encoder_states)
        decoder_dense = Dense(Model.NR_CHARS, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        encoder_input_data, decoder_input_data, decoder_output_data = \
            self.construct_input_output_chars(raw_in=train_raw_in, raw_out=train_raw_out, 
                                                max_size=Model.SRC_TEXT_CHAR_LENGTH)
        if args.verbose:
            print('Coding in/out info')
            for (inn, outs, outc) in zip(encoder_input_data, decoder_input_data, decoder_output_data):
                print('input seq: ', file=log)
                print(inn, file=log)
                print('input shape: ')
                print(inn.shape, file=log)
                print('input (decoder) seq:')
                print(outs, file=log)
                print('input (decoder) shape:')
                print(outs.shape, file=log)
                print('output (train) seq:')
                print(outc, file=log)
                print('output (train) shape:')
                print(outc.shape, file=log)
                break

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=Model.PATIENCE)]
        print(model.summary())
        
        encoder_input_data = np.asarray(encoder_input_data) 
        decoder_input_data = np.asarray(decoder_input_data) 
        decoder_output_data = np.asarray(decoder_output_data)
        print(encoder_input_data.shape, decoder_input_data.shape, decoder_output_data.shape)

        model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
                batch_size=Model.BATCH_SIZE,  epochs=Model.EPOCHS, validation_split=0.2,
                callbacks=callbacks)
        
        """ construct prediction model
            we need to construct a new one because we used teacher forcing (where the input for a timestep is not 
            not the previous generated output, but the correct output) """
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(Model.LATENT_DIM_RNN,))
        decoder_state_input_c = Input(shape=(Model.LATENT_DIM_RNN,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            embedded_decoder, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        test_encoder_input_data, test_deocder_input_data, test_decoder_target_data =\
                self.construct_input_output_chars(raw_in=test_raw_in, raw_out=test_raw_out, 
                                                max_size=Model.SRC_TEXT_CHAR_LENGTH)
        
        """ run on 20 examples from train to see if its okay"""
        for input_seq in encoder_input_data[:20]:
            print(input_seq, file=log)
            self.compute_predictions(input_seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--small_run', dest='small_run', action='store_true', default=False)
    parser.add_argument('--name', dest="name", action="store", default="default")
    #parser.add_argument('--no_chars', dest="no_chars", action="store_true")
    parser.add_argument('--input_file', dest="input_file", action="store", default="infl.csv")
    parser.add_argument('--only_word', dest="only_word", action="store_true", default=False)
    parser.add_argument('--test_file', dest="test_file", action="store", default="test_precision.txt")
    parser.add_argument('--no_train', dest="no_train", action="store_true", default=False)
    parser.add_argument('--no_test', dest="no_test", action="store_true", default=False)
    parser.add_argument('--load', dest="load", action="store", default="infl_detect_all.h5")
    parser.add_argument('--precision_sure', dest="precision_sure", action="store", default=0.8, type=float)
    parser.add_argument('--verbose', dest="verbose", action="store_true", default=False)
    parser.add_argument('--eng_noiser', dest="eng_noiser", action="store_true", default=False)
    parser.add_argument('--beam_width', dest="beam_width", action="store", default=8)
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])
    print('keras version -> {}'.format(keras.__version__))

    model = Model()
    #model.run_model_char_decoder()
    if args.eng_noiser:
        model.eng_noiser()