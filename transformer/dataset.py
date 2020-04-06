import tensorflow as tf
import os
from bert.tokenization.bert_tokenization import FullTokenizer
import tensorflow_datasets as tfds
from typing import Dict, List, Tuple
from transformer.utils import create_masks

def construct_datasets_gec(args, subwords_path):

    if args.bert:
        tokenizer_bert = FullTokenizer(vocab_file=args.bert_model_dir + "vocab.vocab")
        tokenizer_bert.vocab_size = len(tokenizer_bert.vocab)

    examples = get_text_examples_gec(args)

    if os.path.isfile(subwords_path + '.subwords'): 
        tokenizer_ro  = construct_tokenizer_gec(None, subwords_path, args)
    else:
        tokenizer_ro = construct_tokenizer_gec(list(examples), subwords_path, args)

    sample_train = int(args.total_samples * args.train_dev_split)
    gen_dataset = gen_tensors_gec(tokenizer_ro, tokenizer_bert, args)

    dataset = list(gen_dataset)
    dataset = tf.convert_to_tensor(dataset, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    # dataset = tf.data.Dataset.map(prepare_tensors())

    train_dataset = dataset.take(sample_train)
    # train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) # how many batches to prefectch
    # train_dataset = train_dataset.prefetch(1)

    dev_dataset = dataset.skip(sample_train)
    dev_dataset = dev_dataset.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)
    return train_dataset, dev_dataset

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

def gec_generator(tokenizer_ro, tokenizer_bert, args):

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

def gen_tensors_gec(tokenizer_ro, tokenizer_bert, args):
    gen = gec_generator(tokenizer_ro, tokenizer_bert, args)
    for s, t in gen:
        yield (tf.convert_to_tensor(s, dtype=tf.int64), 
                tf.convert_to_tensor(t, dtype=tf.int64))

def get_text_examples_gec(args) -> List[str]:
    gen = gec_generator_text(args)
    return gen

def construct_tokenizer_gec(examples: List, subwords_path, args):

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