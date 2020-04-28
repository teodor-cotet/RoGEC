import tensorflow as tf
import os
from os.path import join
from shutil import copyfile
from bert.tokenization.bert_tokenization import FullTokenizer
import tensorflow_datasets as tfds
from typing import Dict, List, Tuple
import numpy as np
from transformer.utils import create_masks
from transformer.serialization import example_encode_text_dataset, get_text_dataset_tf_records
from transformer.serialization import serialize_ids_dataset

args, tokenizer_ro, tokenizer_bert = None, None, None

train_step_signature_np = [tf.TensorSpec(shape=(None, None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


def construct_flat_datasets(args1, subwords_path):
    global tokenizer_bert, tokenizer_ro, args

    args = args1
    if args.bert:
        tokenizer_bert = FullTokenizer(vocab_file=join(args.bert_model_dir, "vocab.vocab"))
        tokenizer_bert.vocab_size = len(tokenizer_bert.vocab)

    samples = get_text_samples(args)

    if os.path.isfile(subwords_path + '.subwords'): 
        tokenizer_ro  = construct_tokenizer(None, subwords_path, args)
    else:
        tokenizer_ro = construct_tokenizer(list(samples), subwords_path, args)

    sample_train = int(args.total_samples * args.train_dev_split)

    if args.records:
        dataset = tf.data.Dataset.from_generator(generator_tensors_ids_and_segs,
                                        ((tf.int64, tf.int64), tf.int64), 
                                        ((tf.TensorShape([None]), tf.TensorShape([None])), tf.TensorShape([None])))
    else:
        gen_dataset = generator_tensors_ids()
        dataset = list(gen_dataset)
        nr_samples = len(dataset)
        sample_train = int(args.train_dev_split * nr_samples)
        # dataset = tf.convert_to_tensor(dataset, dtype=tf.int64)
        dataset = tf.data.Dataset.from_generator(generator_tensors_ids,
                                        (tf.int64, tf.int64), 
                                        (tf.TensorShape([2, args.seq_length]), tf.TensorShape([args.seq_length])))

    train_dataset = dataset.take(sample_train)
    dev_dataset = dataset.skip(sample_train)

    return train_dataset, dev_dataset

def construct_datasets_gec(args, subwords_path):
    train_dataset, dev_dataset = construct_flat_datasets(args, subwords_path)
    return prepare_datasets(train_dataset, dev_dataset, args)
   
def prepare_datasets(train_dataset, dev_dataset, args):
    train_dataset = train_dataset.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) # how many batches to prefectch

    dev_dataset = dev_dataset.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)
    return train_dataset, dev_dataset

def construct_tf_records(args1, subwords_path=None):
    """given a txt constructs tf records files + subwords dictionary"""
    global tokenizer_bert, tokenizer_ro
    tf.compat.v1.logging.info('constructing tf records files and vocabularies in {}'.format(args1.tf_records))
    train_dataset, dev_dataset = construct_flat_datasets(args1, subwords_path)

    if not os.path.exists(args1.tf_records):
        os.makedirs(args1.tf_records)
    tokenizer_ro_path = os.path.join(args1.tf_records, 'tokenizer_ro')
    tokenizer_bert_path = os.path.join(args1.tf_records, 'tokenizer_bert.vocab')

    tokenizer_ro.save_to_file(tokenizer_ro_path)
    vocab_file_source = join(args1.bert_model_dir, "vocab.vocab")
    if args1.bert:
        copyfile(vocab_file_source, tokenizer_bert_path)
    
    # todo save bert tokenizer
    serialize_ids_dataset(train_dataset, args1, 'train.tfrecord')
    serialize_ids_dataset(dev_dataset, args1, 'dev.tfrecord')
    tf.compat.v1.logging.info('tf records files and vocabularies constructed in {}'.format(args1.tf_records))

def test_map_numpy(tensor1, tensor2):
    global args
    tensor1 = tensor1.numpy()
    tensor1 *= 2 
    t1 = tf.convert_to_tensor(tensor1, dtype=tf.int64)
    tensor1 = tf.reshape(t1, shape=(2, 256))
    tensor2 = tf.reshape(tensor2, shape=(256,))
    return tensor1, tensor2

def construct_datatset_numpy(args1):
    global args
    args = args1
    data1 = tf.random.uniform((15000, 2, 256), maxval=128, dtype=tf.dtypes.int64)
    segs = tf.zeros((15000, 256), dtype=tf.dtypes.int64)

    train_dataset = tf.data.Dataset.from_tensor_slices((data1, segs))
    val_dataset = tf.data.Dataset.from_tensor_slices((data1, segs))
    
    train_dataset = train_dataset.map(lambda t1, t2: tf.py_function(func=test_map_numpy,
        inp=[t1, t2], Tout=(tf.int64, tf.int64)))
    val_dataset = val_dataset.map(lambda t1, t2: tf.py_function(func=test_map_numpy,
        inp=[t1, t2], Tout=(tf.int64, tf.int64)))

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)
    val_dataset = val_dataset.batch(args.batch_size, drop_remainder=True)

    # for x, y in train_dataset.take(2):
    #     print(x.shape)
    return train_dataset, val_dataset

def generator_ids(tokenizer_ro, tokenizer_bert, args):

    with open(args.dataset_file, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                target = line.strip()
            elif i % 2 == 1:
                source = line.strip()
                (source, target), segments = encode_gec(source, target, tokenizer_ro, tokenizer_bert, args)
                if len(source) > args.seq_length or len(target) > args.seq_length:
                    continue
                yield (source, target), segments

def gec_generator_text(args):
    with open(args.dataset_file, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                target = line.strip()
            elif i % 2 == 1:
                source = line.strip()
                yield (source, target)

def make_fixed_length(input_ids: List[int], max_seq_len: int):

    if len(input_ids) < max_seq_len: # pad
        to_add = max_seq_len-len(input_ids)
        for _ in range(to_add):
            input_ids.append(0)
    elif len(input_ids) > max_seq_len: # trim
        input_ids = input_ids[:max_seq_len]
    return input_ids

def encode_gec(source: str, target: str, tokenizer_ro, tokenizer_bert, args):

    if args.bert:
        tokens = ['[CLS]']
        tokens.extend(tokenizer_bert.tokenize(source))
        tokens.append('[SEP]')
        source = tokenizer_bert.convert_tokens_to_ids(tokens)
    else:
        source = [tokenizer_ro.vocab_size] + tokenizer_ro.encode(source) +\
            [tokenizer_ro.vocab_size + 1]
    target = [tokenizer_ro.vocab_size] + tokenizer_ro.encode(target) +\
            [tokenizer_ro.vocab_size + 1]

    segments = [0] * len(source) + [1] * (args.seq_length - len(source))
    source = make_fixed_length(source, args.seq_length)
    target = make_fixed_length(target, args.seq_length)
    segments = make_fixed_length(segments, args.seq_length)

    return (source, target), segments

def generator_tensors_ids():
    global tokenizer_bert, tokenizer_ro, args
    gen = generator_ids(tokenizer_ro, tokenizer_bert, args)
    for step, ((s, t), segs) in enumerate(gen):
        if step % 10000 == 0:
            tf.compat.v1.logging.info('tf record generation, step {}'.format(step))
        s = tf.convert_to_tensor(s, dtype=tf.int64)
        t = tf.convert_to_tensor(t, dtype=tf.int64)
        segs = tf.convert_to_tensor(segs, dtype=tf.int64)
        # print(s.shape, t.shape, segs.shape, tf.stack([s, t]).shape)
        yield tf.stack([s, t]), segs

def generator_tensors_ids_and_segs():
    global tokenizer_bert, tokenizer_ro, args

    gen = generator_ids(tokenizer_ro, tokenizer_bert, args)
    for step, ((s, t), segs) in enumerate(gen):
        if step % 10000 == 0:
            tf.compat.v1.logging.info('tf record generation, step {}'.format(step))
        
        yield (tf.convert_to_tensor(s, dtype=tf.int64), tf.convert_to_tensor(t, dtype=tf.int64)),\
                tf.convert_to_tensor(segs, dtype=tf.int64)
                

def get_text_samples(args) -> List[str]:
    gen = gec_generator_text(args)
    return gen

def construct_tokenizer(examples: List, subwords_path, args):

    if examples is None:
        tokenizer_ro = tfds.features.text.SubwordTextEncoder.load_from_file(subwords_path)
        return tokenizer_ro

    merged_examples = []
    for (source, target) in examples:
        merged_examples.append(source)
        merged_examples.append(target)

    tokenizer_ro = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            merged_examples, target_vocab_size=args.dict_size)
    
    if not os.path.exists(subwords_path):
        os.makedirs(subwords_path)

    tokenizer_ro.save_to_file(subwords_path)

    return tokenizer_ro
