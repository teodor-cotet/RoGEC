import tensorflow as tf
import os
from shutil import copyfile
from bert.tokenization.bert_tokenization import FullTokenizer
import tensorflow_datasets as tfds
from typing import Dict, List, Tuple
import numpy as np
from transformer.utils import create_masks
from transformer.serialization import example_encode_text_dataset, get_text_dataset_tf_records
from transformer.serialization import serialize_ids_dataset

args, tokenizer_ro, tokenizer_bert = None, None, None

def construct_flat_datasets(args, subwords_path):
    global tokenizer_bert, tokenizer_ro
    if args.bert:
        tokenizer_bert = FullTokenizer(vocab_file=args.bert_model_dir + "vocab.vocab")
        tokenizer_bert.vocab_size = len(tokenizer_bert.vocab)

    samples = get_text_samples(args)

    if os.path.isfile(subwords_path + '.subwords'): 
        tokenizer_ro  = construct_tokenizer(None, subwords_path, args)
    else:
        tokenizer_ro = construct_tokenizer(list(samples), subwords_path, args)

    sample_train = int(args.total_samples * args.train_dev_split)
    gen_dataset = generator_tensors_ids(tokenizer_ro, tokenizer_bert, args)

    dataset = list(gen_dataset)
    nr_samples = len(dataset)
    dataset = tf.convert_to_tensor(dataset, dtype=tf.int64)
    segs = tf.zeros((nr_samples, args.seq_length), dtype=tf.dtypes.int64)
    dataset = tf.data.Dataset.from_tensor_slices((dataset, segs))

    train_dataset = dataset.take(sample_train)
    dev_dataset = dataset.skip(sample_train)

    return train_dataset, dev_dataset

def construct_datasets_gec(args, subwords_path):
    train_dataset, dev_dataset = construct_flat_datasets(args, subwords_path)

    train_dataset = train_dataset.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) # how many batches to prefectch

    dev_dataset = dev_dataset.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)
    return train_dataset, dev_dataset

def construct_tf_records(args1, subwords_path=None):
    """given a txt constructs tf records files + subwords dictionary"""
    global tokenizer_bert, tokenizer_ro
    train_dataset, dev_dataset = construct_flat_datasets(args1, subwords_path)

    if not os.path.exists(args1.tf_records):
        os.makedirs(args1.tf_records)
    tokenizer_ro_path = os.path.join(args1.tf_records, 'tokenizer_ro')
    tokenizer_bert_path = os.path.join(args1.tf_records, 'tokenizer_bert.vocab')

    tokenizer_ro.save_to_file(tokenizer_ro_path)
    vocab_file_source = args1.bert_model_dir + "vocab.vocab"
    copyfile(vocab_file_source, tokenizer_bert_path)
    
    # todo save bert tokenizer
    serialize_ids_dataset(train_dataset, args1, 'train.tfrecord')
    serialize_ids_dataset(dev_dataset, args1, 'dev.tfrecord')

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
    # data1 = np.random.randint(100, size=(1024, 128))
    #data1 = tf.convert_to_tensor(data1, dtype=tf.int64)

    # data2 = np.random.randint(100, size=(1024, 128))
    # data2 = tf.convert_to_tensor(data2, dtype=tf.int64)

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

def prepare_tensors(inp, tar):
    # inp, tar with shape = (seq_length, )
    segs = tf.ones((64, 38), dtype=tf.int64)
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    return (inp, segs, enc_padding_mask, combined_mask, dec_padding_mask), tar

def make_dsitr(source: List[int], tar: List[int]):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(source, tar_inp)

def generator_ids(tokenizer_ro, tokenizer_bert, args):

    with open(args.dataset_file, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                target = line.strip()
            elif i % 2 == 1:
                source = line.strip()
                source, target = encode_gec(source, target, tokenizer_ro, tokenizer_bert, args)
                if len(source) > args.seq_length or len(target) > args.seq_length:
                    continue
                yield (source, target)

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
        # target = [tokenizer_bert.vocab_size] + tokenizer_bert.convert_tokens_to_ids(tokenizer_bert.tokenize(target)) +\
        #         [tokenizer_bert.vocab_size + 1]
    else:
        source = [tokenizer_ro.vocab_size] + tokenizer_ro.encode(source) +\
            [tokenizer_ro.vocab_size + 1]
    target = [tokenizer_ro.vocab_size] + tokenizer_ro.encode(target) +\
            [tokenizer_ro.vocab_size + 1]
    
    source = make_fixed_length(source, args.seq_length)
    target = make_fixed_length(target, args.seq_length)
  
    return source, target

def generator_tensors_ids(tokenizer_ro, tokenizer_bert, args):
    gen = generator_ids(tokenizer_ro, tokenizer_bert, args)
    for s, t in gen:
        yield (tf.convert_to_tensor(s, dtype=tf.int64), 
                tf.convert_to_tensor(t, dtype=tf.int64))

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
