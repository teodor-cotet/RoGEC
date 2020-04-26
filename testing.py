
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import time
from typing import Dict, List, Tuple
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app as absl_app
from bert.tokenization.bert_tokenization import FullTokenizer

from transformer.dataset import construct_datasets_gec, construct_tokenizer, prepare_tensors,\
        construct_datatset_numpy, prepare_datasets
from transformer.utils import create_masks
from transformer.transformer_bert import TransformerBert
from transformer.transformer import Transformer
from transformer.transformer_scheduler import CustomSchedule
from transformer.dataset import construct_tf_records
from transformer.serialization import get_ids_dataset_tf_records

tf.compat.v1.flags.DEFINE_bool("use_map", True, "")
tf.compat.v1.flags.DEFINE_bool("custom", True, "")

# TPU cloud params
tf.compat.v1.flags.DEFINE_string(
    "tpu", default='teodor-cotet',
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.compat.v1.flags.DEFINE_string(
    "tpu_zone", default='us-central1-f',
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.compat.v1.flags.DEFINE_string(
    "gcp_project", default='rogec-271608',
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.compat.v1.flags.DEFINE_bool("use_tpu", False, "Use TPUs rather than plain CPUs")
tf.compat.v1.flags.DEFINE_bool("test", False, "Use TPUs rather than plain CPUs")
tf.compat.v1.flags.DEFINE_string('bucket', default='ro-gec', help='path from where to load bert')


# paths for model  1k_clean_dirty_better.txt 30k_clean_dirty_better.txt
tf.compat.v1.flags.DEFINE_string('dataset_file', default='corpora/synthetic_wiki/1k_clean_dirty_better.txt', help='')
tf.compat.v1.flags.DEFINE_string('checkpoint', default='checkpoints/transformer_test',
                help='Checpoint save locations, or restore')
# tf.compat.v1.flags.DEFINE_string('subwords', default='checkpoints/transformer_test/corpora', help='')
tf.compat.v1.flags.DEFINE_string('bert_model_dir', default='./bert/ro0/', help='path from where to load bert')
tf.compat.v1.flags.DEFINE_string('tf_records', default='./corpora/tf_records/test/', help='path to tf records folder')

# mode of execution
"""if bert is used, the decoder is still a transofrmer with transformer specific tokenization"""
tf.compat.v1.flags.DEFINE_bool('bert', default=False, help='use bert as encoder or transformer')
tf.compat.v1.flags.DEFINE_bool('train_mode', default=False, help='do training')
tf.compat.v1.flags.DEFINE_bool('decode_mode',default=False, help='do prediction, decoding')

# model params
tf.compat.v1.flags.DEFINE_integer('num_layers', default=6, help='')
tf.compat.v1.flags.DEFINE_integer('d_model', default=256,
                        help='d_model size is the out of the embeddings, it must match the bert model size, if you use one')
tf.compat.v1.flags.DEFINE_integer('seq_length', default=256, help='same as d_model')
tf.compat.v1.flags.DEFINE_integer('dff', default=256, help='')
tf.compat.v1.flags.DEFINE_integer('num_heads', default=8, help='')
tf.compat.v1.flags.DEFINE_float('dropout', default=0.1, help='')
tf.compat.v1.flags.DEFINE_integer('dict_size', default=(2**9), help='')
tf.compat.v1.flags.DEFINE_integer('epochs', default=10, help='')
tf.compat.v1.flags.DEFINE_integer('buffer_size', default=(100), help='')
tf.compat.v1.flags.DEFINE_integer('batch_size', default=32, help='')
tf.compat.v1.flags.DEFINE_integer('max_length', default=256, help='')
tf.compat.v1.flags.DEFINE_float('train_dev_split', default=0.9, help='')
tf.compat.v1.flags.DEFINE_integer('total_samples', default=500, help='')
tf.compat.v1.flags.DEFINE_bool('show_batch_stats', default=True, help='do prediction, decoding')

# for prediction purposes only
tf.compat.v1.flags.DEFINE_string('in_file_decode', default='corpora/cna/dev_old/small_decode_test.txt', help='')
tf.compat.v1.flags.DEFINE_string('out_file_decode', default='corpora/cna/dev_predicted_2.txt', help='')

args = tf.compat.v1.flags.FLAGS


args = tf.compat.v1.flags.FLAGS

class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        # self.inp1 = tf.keras.Input(shape=(1023,))
        # self.inp2 = tf.keras.Input(shape=(1023,))
        self.con = tf.keras.layers.Concatenate()
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(2)
    
    def call(self, inputs):
        x = self.con(inputs)
        x = self.d1(x)
        return self.d2(x)

def create_model():

    
    if args.custom:
        model = TestModel([inp1, inp2])
    else:
        inp1 = tf.keras.Input(shape=(1023,))
        inp2 = tf.keras.Input(shape=(1023,))
        x = tf.keras.layers.Concatenate()([inp1, inp2])
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        y = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs=[inp1, inp2], outputs=y)
    return model

def get_dataset(batch_size=200):
  datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True,
                             try_gcs=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0

    return image, label

  train_dataset = mnist_train.map(scale).shuffle(10000).batch(batch_size)
  test_dataset = mnist_test.map(scale).batch(batch_size)

  return train_dataset, test_dataset

def scale_funct(d1, d2, label):
    d1 /= 2.0
    d2 /= 2.0
    d1 = d1[1:]
    d2 = d2[1:]
    return (d1, d2), label

def get_custom_dataset(total_samples, batch_size):
    global args
    data1 = np.random.uniform(.0, 2.0, (total_samples, 1024))
    data1 = tf.convert_to_tensor(data1, dtype=tf.float32)

    data2 = np.random.uniform(.0, 2.0, (total_samples, 1024))
    data2 = tf.convert_to_tensor(data2, dtype=tf.float32)

    labels = np.random.randint(2, size=(total_samples,))
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((data1, data2, labels))
    
    if args.use_map:
        train_dataset = train_dataset.map(scale_funct)
    train_dataset = train_dataset.repeat(5).batch(batch_size, drop_remainder=True)
    for x1, x2 in train_dataset.take(1):
        print(x1[0].shape, x2.shape)
    return train_dataset


def generator():
    global args
    for _ in range(args.samples):
        data = np.random.uniform(.0, 2.0, (64, 64, 1))
        data = tf.convert_to_tensor(data, dtype=tf.float32)

        label = np.random.randint(10)
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        yield (data, label)

def get_generator_dataset(total_samples, batch_size):
    global args
    train_dataset = tf.data.Dataset.from_generator(generator, 
                                                    output_types=(tf.float32, tf.int32), 
                                                    output_shapes=(tf.TensorShape([64, 64, 1]), tf.TensorShape([])))
    if args.use_map:
        train_dataset = train_dataset.map(scale_funct)
    train_dataset = train_dataset.repeat(5).batch(batch_size, drop_remainder=True)
    return train_dataset

def get_tfrecord_dataset(total_samples, batch_size):
    pass

def test_dataset():
    construct_tf_records(args, subwords_path)
    train_dataset, dev_dataset, tokenizer_ro, tokenizer_bert = get_ids_dataset_tf_records(args)

    for x, y in dev_dataset.take(1):
        source = x[0].numpy()
        target = x[1].numpy()
        print(source)
        source = [x for x in source if x < args.dict_size]
        target = [x for x in target if x < args.dict_size]
        source = tokenizer_ro.decode(source)
        target = tokenizer_ro.decode(target)
        print(source)
        print(target)

def test_bert_trans():
    if args.bert is True:
        sample_transformer = TransformerBert(num_layers=2, d_model=512, num_heads=8, dff=2048, 
            input_vocab_size=8500, target_vocab_size=8000, 
            model_dir=args.bert_model_dir, pe_input=10000, pe_target=6000)
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

    tf.compat.v1.logging.info(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
    
def main(argv):
    del argv
    global args
    batch_size = args.batch
    total_samples = args.samples

    if args.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

        with strategy.scope():
            model = create_model()
            model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['sparse_categorical_accuracy'])
        print(model.summary())
        print(model.count_params())

        train_dataset = get_custom_dataset(total_samples, batch_size)
        model.fit(train_dataset, epochs=5, steps_per_epoch=total_samples//batch_size)
    else:
        model = create_model()
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['sparse_categorical_accuracy'])
        print(model.summary())
        print(model.count_params())
        train_dataset = get_custom_dataset(total_samples, batch_size)
        model.fit(train_dataset, epochs=5, steps_per_epoch=total_samples//batch_size)


if __name__ == "__main__":
    absl_app.run(main)