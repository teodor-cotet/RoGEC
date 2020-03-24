
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
from bert.tokenization.bert_tokenization import FullTokenizer

import time
import numpy as np
import matplotlib.pyplot as plt
import argparse, os
from typing import Tuple, List, Dict
import bert
from bert.tokenization.bert_tokenization import FullTokenizer

BUFFER_SIZE = 20000
BATCH_SIZE = 8
MAX_LENGTH = 252
TRAIN_DEV_SPLIT = 0.9
TOTAL_SAMPLES = 10e6
EPOCHS = 100
D_MODEL = 256
tokenizer_pt, tokenizer_en, tokenizer_ro, tokenizer_bert = None, None, None, None
args = None
transformer, optimizer, train_loss, train_accuracy = None, None, None, None
eval_loss, eval_accuracy = None, None
train_step_signature = [
        tf.TensorSpec(shape=(None, D_MODEL), dtype=tf.int64),
        tf.TensorSpec(shape=(None, D_MODEL), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]
eval_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]


"Add a start and end token to the input and target."
def encode(lang1, lang2):
    global tokenizer_pt, tokenizer_en
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size+1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size+1]

    return lang1, lang2

"""Drop examples with > MAX_LENGTH """
def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def tf_encode(pt, en):
  return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
        
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32) # returns 0 and 1 float
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) # select band diagonal -> lower this case
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)

def example_tokenizer(tokenizer_en, tokenizer_pt):
    # example tokenizer
    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer_en.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))
    original_string = tokenizer_en.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))
    assert original_string == sample_string
    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

def example_scaled_dot_attention():
    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        [0,0,10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[   1,0],
                        [  10,0],
                        [ 100,5],
                        [1000,6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

def example_pos_encoding():
    for i in range(20):
        pos_encoding = positional_encoding(i, 512)
        print(pos_encoding)
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

def example_padding_mask():
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    print(create_padding_mask(x))

def example_multihead_attention():
    """ At each location in the sequence, y,
     the MultiHeadAttention runs all 8 attention heads across all other locations in the sequence,
      returning a new vector of the same length at each location."""

    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape)
    print(attn.shape)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        """ split q, k, v by nr of heads """ 
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights

def example_point_wise_nn():
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)

def point_wise_feed_forward_network(d_model, dff):
    """point wise nn because if u have (seq_length, hidden) at each seq_length you multiple 
        by a the same dense layer. this is the default behaviour of a dense layer in keras anyways """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, 
            look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
            
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, model_dir, d_model):
        super(BertEncoder, self).__init__()
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")
            
    def call(self, input_ids, segment_ids, training):
        bert_output = self.bert_layer([input_ids, segment_ids])
        return bert_output  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                look_ahead_mask, padding_mask)
        
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class BertModel:

    def __init__(self, model_dir, max_seq_len):

        self.model_dir = model_dir
        self.tokenizer = FullTokenizer(vocab_file=model_dir+"vocab.vocab")
        self.max_seq_len = max_seq_len
        self.bert_layer = self.load_bert()

    def load_bert(self):
        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")
        return bert_layer

    def __get_segments(self, tokens):
        sep_enc = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == sep_enc[0]:
                if current_segment_id == 0:
                    current_segment_id = 1
        return segments

    # use it with sentence2=None for single sentence
    def process_sentences(self, sentence1, sentence2=None):
        tokens = ['[CLS]']
        tokens.extend(self.tokenizer.tokenize(sentence1))
        tokens.append('[SEP]')
        if sentence2 != None:
            tokens.extend(self.tokenizer.tokenize(sentence2))
            tokens.append('[SEP]')

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1] * len(input_ids)
        input_segments = self.__get_segments(input_ids)

        if self.max_seq_len == None:
            input_ids = np.array(input_ids)
            input_masks = np.array(input_masks)
            input_segments = np.array(input_segments)
            
            return input_ids, input_masks, input_segments

        else: # pad or trim
            if len(input_ids) < self.max_seq_len: # pad
                to_add = self.max_seq_len-len(input_ids)
                for _ in range(to_add):
                    input_ids.append(0)
                    input_masks.append(0)
                    input_segments.append(0)

            elif len(input_ids) > self.max_seq_len: # trim
                input_ids = input_ids[:self.max_seq_len]
                input_masks = input_masks[:self.max_seq_len]
                input_segments = input_segments[:self.max_seq_len]

            input_ids = np.array(input_ids)
            input_masks = np.array(input_masks)
            input_segments = np.array(input_segments)

            return input_ids, input_masks, input_segments

class TransformerBert(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                target_vocab_size, model_dir, pe_input, pe_target, rate=0.1):
        super(TransformerBert, self).__init__()

        self.encoder = BertEncoder(model_dir=model_dir, d_model=d_model)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                            target_vocab_size, pe_target, rate)
        # self.segments = np.zeros((d_model,))
        # self.ids = np.ones((d_model,))
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, input_ids, input_seg, tar, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_ids, input_seg, training)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                            input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                            target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    # mask to compute loss
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

@tf.function(input_signature=eval_step_signature)
def eval_step(inp, tar):
    global transformer, optimizer, eval_loss, eval_accuracy
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions)
    eval_loss(loss)
    eval_accuracy(tar_real, predictions)

@tf.function(input_signature=train_step_signature)
def train_step(inp, inp_seg, tar):
    global transformer, optimizer, train_loss, train_accuracy
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    #print(inp.shape, inp_seg.shape, tar.shape)
    # temp_input = tf.random.uniform((64, 256), dtype=tf.int64, minval=0, maxval=200)
    # temp_seg = tf.ones((64, 256), dtype=tf.int64)
    # temp_target = tf.random.uniform((64, 254), dtype=tf.int64, minval=0, maxval=200)
    
    with tf.GradientTape() as tape:
        if args.bert is True:
            predictions, _ = transformer(inp, inp_seg, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        else:
            predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

def gec_generator():
    global args
    print(args.dataset_file)
    with open(args.dataset_file, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                target = line.strip()
            elif i % 2 == 1:
                source = line.strip()
                yield (source, target)

def filter_max_length_gec(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def encode_gec(source: str, target: str):
    global tokenizer_ro, tokenizer_bert
    if args.bert:
        tokens = ['[CLS]']
        tokens.extend(tokenizer_bert.tokenize(source.numpy()))
        tokens.append('[SEP]')
        source = tokenizer_bert.convert_tokens_to_ids(tokens)
    else:
        source = [tokenizer_ro.vocab_size] + tokenizer_ro.encode(source.numpy()) +\
            [tokenizer_ro.vocab_size + 1]
    target = [tokenizer_ro.vocab_size] + tokenizer_ro.encode(target.numpy()) +\
         [tokenizer_ro.vocab_size + 1]
    return source, target

def tf_encode_gec(source: str, target: str):
    return tf.py_function(func=encode_gec, inp=[source, target], Tout=[tf.int64, tf.int64])

def gen_tensors_gec():
    gen = gec_generator()
    for s, t in gen:
        yield (tf.convert_to_tensor(s, dtype=tf.string), 
                tf.convert_to_tensor(t, dtype=tf.string))

def get_examples_gec() -> List[str]:
    global args, tokenizer_ro
    gen = gec_generator()
    return gen

def construct_subwords_gec(examples: List):
    global args, tokenizer_bert, tokenizer_ro
    if args.bert:
        tokenizer_bert = FullTokenizer(vocab_file=args.bert_model_dir + "vocab.vocab")
    if examples is None:
        tokenizer_ro = tfds.features.text.SubwordTextEncoder.load_from_file(args.subwords)
        return tokenizer_ro

    merged_examples = []
    for (source, target) in examples:
        merged_examples.append(source)
        merged_examples.append(target)

    tokenizer_ro = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            merged_examples, target_vocab_size=args.dict_size)
    
    if not os.path.exists(args.subwords):
        os.makedirs(args.subwords)

    tokenizer_ro.save_to_file(args.subwords)

    return tokenizer_ro

def construct_datasets_gec():
    global args, tokenizer_ro, TRAIN_DEV_SPLIT, TOTAL_SAMPLES
    examples = get_examples_gec()
    if os.path.isfile(args.subwords + '.subwords'):
        tokenizer_ro  = construct_subwords_gec(None)
        print('subwords restored')
    else:
        tokenizer_ro  = construct_subwords_gec(list(examples))

    sample_train = int(TOTAL_SAMPLES * TRAIN_DEV_SPLIT)

    dataset = tf.data.Dataset.from_generator(gen_tensors_gec, output_types=(tf.string, tf.string))
    dataset = dataset.map(tf_encode_gec)
    train_dataset = dataset.take(sample_train)
    train_dataset = train_dataset.filter(filter_max_length_gec)
    # train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=([args.seq_length], [-1])) # pad with 0
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) # how many batches to prefectch
    # train_dataset = train_dataset.prefetch(1)

    dev_dataset = dataset.skip(sample_train)
    dev_dataset = dev_dataset.filter(filter_max_length_gec)
    dev_dataset = dev_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=([args.seq_length], [-1]))
    return train_dataset, dev_dataset

def generate_sentence_gec(inp_sentence: str):
    global tokenizer_ro, transformer, optimizer, args, MAX_LENGTH

    if tokenizer_ro is None:
        if os.path.isfile(args.subwords + '.subwords'):
            print('Vocabulary loaded\n')
            tokenizer_ro  = construct_subwords_gec(None)
        else:
            examples = get_examples_gec()
            tokenizer_ro  = construct_subwords_gec(examples)

    if transformer is None:
        transformer, optimizer = get_model_gec()
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            # print(optimizer._decayed_lr(tf.float32))
            # loading mechanis matches variables from the tf graph and resotres their values
            ckpt.restore(ckpt_manager.latest_checkpoint)
        else:
            print('No checkpoints for transformers. Aborting')
            return None

    start_token = [tokenizer_ro.vocab_size]
    end_token = [tokenizer_ro.vocab_size + 1]

    inp_sentence = start_token + tokenizer_ro.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_ro.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_ro.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights_gec(attention, sentence, result, layer):
    global tokenizer_ro
    fig = plt.figure(figsize=(16, 8))
    sentence = tokenizer_ro.encode(sentence)
    attention = tf.squeeze(attention[layer], axis=0)
    
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
        
        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}
        
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))
        
        ax.set_ylim(len(result)-1.5, -0.5)
            
        ax.set_xticklabels(
            ['<start>']+[tokenizer_ro.decode([i]) for i in sentence]+['<end>'], 
            fontdict=fontdict, rotation=90)
        
        ax.set_yticklabels([tokenizer_ro.decode([i]) for i in result 
                            if i < tokenizer_ro.vocab_size], 
                        fontdict=fontdict)
        
        ax.set_xlabel('Head {}'.format(head+1))
    
    plt.tight_layout()
    plt.show()

def correct_from_file(in_file: str, out_file: str):
    with open(in_file, 'r') as fin, open(out_file, 'w') as fout:
        for line in fin:
            print(line)
            predicted_sentences = correct_gec(line)
            #fout.write(line)
            fout.write(predicted_sentences + '\n')
            fout.flush()

def correct_gec(sentence: str, plot=''):
    global tokenizer_ro
    result, attention_weights = generate_sentence_gec(sentence)
    predicted_sentence = tokenizer_ro.decode([i for i in result 
                                                if i < tokenizer_ro.vocab_size])  
    # print('Input: {}'.format(sentence))
    # print('Predicted sentence: {}'.format(predicted_sentence))
    
    if plot:
        plot_attention_weights_gec(attention_weights, sentence, result, plot)
    return predicted_sentence
     
def get_model_gec():
    global args, transformer, tokenizer_ro

    vocab_size = tokenizer_ro.vocab_size + 2

    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
    if args.bert is True:
        transformer = TransformerBert(args.num_layers, args.d_model, args.num_heads, args.dff,
                            vocab_size, vocab_size,
                            model_dir='./bert/teo/ro0/', 
                            pe_input=vocab_size, 
                            pe_target=vocab_size,
                            rate=args.dropout)
        print('transformer bert loaded')
    else:
        transformer = Transformer(args.num_layers, args.d_model, args.num_heads, args.dff,
                            vocab_size, vocab_size, 
                            pe_input=vocab_size, 
                            pe_target=vocab_size,
                            rate=args.dropout)
    return transformer, optimizer

def train_gec():
    global args, optimizer, transformer, train_loss, train_accuracy, eval_loss, eval_accuracy, BATCH_SIZE
    with open('run.txt', 'wt') as log:
        
        train_dataset, dev_dataset = construct_datasets_gec()
        
        # for x, y in train_dataset.take(5):
        #     print(x)
        #     print(x.shape)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')

        transformer, optimizer = get_model_gec()
        # object you want to checkpoint are saved as attributes of the chepoint obj
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            # print(optimizer._decayed_lr(tf.float32))
            # loading mechanis matches variables from the tf graph and resotres their values
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            # print(optimizer._decayed_lr(tf.float32))

        # train
        for epoch in range(args.epochs):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()
            eval_loss.reset_states()
            eval_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                inp_seg = tf.zeros(shape=inp.shape, dtype=tf.dtypes.int64)
                train_step(inp, inp_seg, tar)
                if batch % 5000 == 0:
                    print('train - epoch {} batch {} loss {:.4f} accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                    log.write('train - epoch {} batch {} loss {:.4f} accuracy {:.4f}\n'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                    log.flush()

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                log.write('Saving checkpoint for epoch {} at {} \n'.format(epoch+1,
                                                                    ckpt_save_path))
                log.flush()
            
            print('Final train - epoch {} loss {:.4f} accuracy {:.4f}'.format(epoch + 1, 
                                                            train_loss.result(), 
                                                            train_accuracy.result()))
            log.write('Final train - epoch {} loss {:.4f} accuracy {:.4f} \n'.format(epoch + 1, 
                                                            train_loss.result(), 
                                                            train_accuracy.result()))
            log.flush()

            for (batch, (inp, tar)) in enumerate(dev_dataset):
                eval_step(inp, tar)
                if batch % 1000 == 0:
                    print('Dev - epoch {} batch {} loss {:.4f} accuracy {:.4f}'.format(
                        epoch + 1, batch, eval_loss.result(), eval_accuracy.result()))
                    log.write('Dev - epoch {} batch {} loss {:.4f} accuracy {:.4f}\n'.format(
                        epoch + 1, batch, eval_loss.result(), eval_accuracy.result()))
                    log.flush()
                    
            print('Final dev - epoch {} batch {} loss {:.4f} accuracy {:.4f}'.format(
                        epoch + 1, batch, eval_loss.result(), eval_accuracy.result()))
            log.write('Final dev - epoch {} batch {} loss {:.4f} accuracy {:.4f}\n'.format(
                        epoch + 1, batch, eval_loss.result(), eval_accuracy.result()))
            log.flush()

def test_bert_trans():
    if args.bert is True:
        sample_transformer = TransformerBert(num_layers=2, d_model=512, num_heads=8, dff=2048, 
            input_vocab_size=8500, target_vocab_size=8000, 
            model_dir='./bert/teo/ro0/', pe_input=10000, pe_target=6000)
    else:
        sample_transformer = Transformer(
            num_layers=2, d_model=512, num_heads=8, dff=2048, 
            input_vocab_size=8500, target_vocab_size=8000, 
            pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_seg = tf.ones((64, 38), dtype=tf.int64)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(temp_input, temp_target)
    if args.bert is True:
        fn_out, _ = sample_transformer(temp_input, temp_seg, temp_target, training=True, 
                                    enc_padding_mask=enc_padding_mask, 
                                    look_ahead_mask=combined_mask,
                                    dec_padding_mask=dec_padding_mask)
    else:
        fn_out, _ = sample_transformer(temp_input, temp_target, training=False, 
                                    enc_padding_mask=None, 
                                    look_ahead_mask=None,
                                    dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dataset_file', dest='dataset_file', action="store",
                         default='corpora/synthetic_wiki/1k_clean_dirty_better.txt')

    parser.add_argument('-checkpoint', dest='checkpoint', action="store", default='checkpoints/transformer_test')
    parser.add_argument('-subwords', dest='subwords', action="store", default='checkpoints/transformer_test/corpora')
    parser.add_argument('-bert_model_dir', dest='bert_model_dir', action="store", default='./bert/teo/ro0/')
    
    parser.add_argument('-bert', dest='bert', action="store_true", default=False)
    parser.add_argument('-train_mode', dest='train_mode', action="store_true", default=False)
    parser.add_argument('-decode_mode', dest='decode_mode', action="store_true", default=False)

    """model params"""
    parser.add_argument('-num_layers', dest='num_layers', action="store", type=int, default=6)
    # d_model size is the out of the embeddings, it must match the bert model size, if you use one
    parser.add_argument('-d_model', dest='d_model', action="store", type=int, default=256)
    parser.add_argument('-dff', dest='dff', action="store", type=int, default=256)
    parser.add_argument('-num_heads', dest='num_heads', action="store", type=int, default=8)
    parser.add_argument('-dropout', dest='dropout', action="store", type=float, default=0.1)
    parser.add_argument('-dict_size', dest='dict_size', action="store", type=int, default=(2**15))
    parser.add_argument('-epochs', dest='epochs', action="store", type=int, default=100)
    parser.add_argument('-seq_length', dest='seq_length', action="store", type=int, default=256)

    # test stuff
    parser.add_argument('-in_file_decode', dest='in_file_decode', action="store", default='corpora/cna/dev_old/small_decode_test.txt')
    parser.add_argument('-out_file_decode', dest='out_file_decode', action="store", default='corpora/cna/dev_predicted_2.txt')
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])

    # example_pos_encoding()
    # main()
    # example_scaled_dot_attention()
    # example_multihead_attention()
    # example_point_wise_nn()
    # construct_dataset_gec()
    if args.train_mode:
        # test_bert_trans()
        train_gec()
    if args.decode_mode:
        correct_from_file(in_file=args.in_file_decode, out_file=args.out_file_decode)