import os
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bert.tokenization.bert_tokenization import FullTokenizer
from google.cloud import storage

args = None

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _tensor_feature(value):
    """converts a tensor to serialized byte string"""
    return  _bytes_feature(tf.io.serialize_tensor(value))

def serialize_example(data, seg):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'data': _tensor_feature(data),
        'segs': _tensor_feature(seg),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_example_text(s, t):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'source': _bytes_feature(s),
        'target': _bytes_feature(t),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_example_ids(sentences, seg):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'sentences': _tensor_feature(sentences),
        'seg': _tensor_feature(seg),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example_ids(sentences, seg):
    tf_string = tf.py_function(
        serialize_example_ids,
        (sentences, seg),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar

def tf_serialize_example(data, seg):
    tf_string = tf.py_function(
        serialize_example,
        (data, seg),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar

def example_encode_tensor():
    x = np.random.randint(0, high=128, size=(4, 2, 8))
    xx = np.random.randint(0, high=128, size=(4, 8))
    x = tf.convert_to_tensor(x, dtype=tf.int32)
    xx = tf.convert_to_tensor(xx, dtype=tf.int32)

    serialized_example = serialize_example(x, xx) # tensor ->  byte string -> tf.feature -> tf example -> serialized string
    example_proto = tf.train.Example.FromString(serialized_example) # serialized string -> tf example 
    print(example_proto)
    feature = example_proto.features.feature # tf example -> tf feature
    print(feature)
    segs = feature['segs'].bytes_list.value[0] # tf feature -> byte string (serialized 2) 
    print(segs)
    y = tf.io.parse_tensor(segs, out_type=tf.int32) # byte string -> tensor 
    print(y)
  
def example_encode_text():

    s = "asdqăđșßâ".encode('utf-8')
    t = "ăđßr€rțhy".encode('utf-8')

    serialized_example = serialize_example_text(s, t) # text ->  -> tf.feature -> tf example -> serialized string
    example_proto = tf.train.Example.FromString(serialized_example) # serialized string -> tf example 
    print(example_proto)
    feature = example_proto.features.feature # tf example -> tf feature
    print(feature)
    source = feature['source'].bytes_list.value[0] # tf feature -> text
    print(source.decode('utf-8'))

def generator_text():
    for _ in range(1024):
        s = 'asda'.encode('utf-8')
        t = 'asda'.encode('utf-8')
        yield serialize_example_text(s, t)

def parse_example(example):
    feature_description = {
        'source': tf.io.VarLenFeature(tf.string), # or tf.fixedLen
        'target': tf.io.VarLenFeature(tf.string)
    }
    y = tf.io.parse_single_example(example, feature_description) # get the tensor
    return (tf.sparse.to_dense(y['source'])[0], tf.sparse.to_dense(y['target'])[0])

def parse_example_ids(example):
    global args 
    # todo
    feature_description = {
        'sentences': tf.io.FixedLenFeature((), tf.string), # or tf.fixedLen
        'seg': tf.io.FixedLenFeature((), tf.string)
    }
    parsed_example = tf.io.parse_single_example(example, feature_description) # get the tensor

    seg = parsed_example['seg']
    seg = tf.io.parse_tensor(seg, out_type=tf.int64)
    seg = tf.reshape(seg, shape=(args.seq_length, ))

    sentences = parsed_example['sentences']
    sentences = tf.io.parse_tensor(sentences, out_type=tf.int64)
    sentences = tf.reshape(sentences, shape=(2, args.seq_length))
    return sentences, seg

def example_encode_text_dataset(args, filename='test.tfrecord'):
    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator_text, output_types=tf.string, output_shapes=())
    # write the dataset to a file
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)
    # read from tf record file
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    # for x in raw_dataset.take(5):
    #     print(x)
    dataset = raw_dataset.map(parse_example)

        #print(x[0].numpy()[0].decode('utf-8'), x[1].numpy()[0].decode('utf-8'))
    return dataset

def serialize_ids_dataset(dataset, args, filename='train.tfrecord'):
    tf_records_path = os.path.join(args.tf_records, filename)
    serialized_dataset = dataset.map(tf_serialize_example_ids)
    writer = tf.data.experimental.TFRecordWriter(tf_records_path)
    writer.write(serialized_dataset)

def get_ids_dataset_tf_records(args1):
    global args
    args = args1
    # get dataset
    path_tf_records = args.tf_records
    if args.use_tpu:
        path_tf_records = join('gs://', args.bucket, path_tf_records)

    train_tf_record_file = join(path_tf_records, 'train.tfrecord')
    dev_tf_record_file = join(path_tf_records, 'dev.tfrecord')

    tf.compat.v1.logging.info('restoring tf records from {} {}'.format(train_tf_record_file, dev_tf_record_file))

    raw_train_dataset = tf.data.TFRecordDataset(train_tf_record_file)
    train_dataset = raw_train_dataset.map(parse_example_ids)

    raw_dev_dataset = tf.data.TFRecordDataset(dev_tf_record_file)
    dev_dataset = raw_dev_dataset.map(parse_example_ids)
    return train_dataset, dev_dataset

def get_tokenizers_ckeckpoint(args1):
    global args
    args = args1
    tokenizer_ro_path = join(args.checkpoint, 'tokenizer_ro')
    tokenizer_ro = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_ro_path)
    tf.compat.v1.logging.info('restoring ro tokenizer from {}'.format(tokenizer_ro_path))

    tokenizer_bert = None
    if args.bert:
        tokenizer_bert_path = join(args.checkpoint, 'tokenizer_bert.vocab')
        tokenizer_bert = FullTokenizer(vocab_file=tokenizer_bert_path)
        tokenizer_bert.vocab_size = len(tokenizer_bert.vocab)
        tf.compat.v1.logging.info('restoring bert tokenizer from {}'.format(tokenizer_bert_path))

    tf.compat.v1.logging.info('tokenizers restored')
    return tokenizer_ro, tokenizer_bert 


def get_text_dataset_tf_records(path_tf_records):
    tf_records_files = [join(path_tf_records, f) for f in listdir(path_tf_records) \
            if isfile(join(path_tf_records, f)) and f.endswith('.tfrecord')]
    
    raw_dataset = tf.data.TFRecordDataset(tf_records_files)
    dataset = raw_dataset.map(parse_example)

    return dataset

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    tf.compat.v1.logging.info('file {} uploaded {} to bucket {}'.format(
        source_file_name, destination_blob_name, bucket_name))

if __name__ == "__main__":
   
    # example_encode_text()
    # example_encode_text_dataset(None)
    dataset = get_text_dataset_tf_records('./corpora/tf_records/test/')
    # for x, y in dataset:
    #     print(x)
    examples = [(s.numpy(), t.numpy()) for s, t in dataset]
    # for x, y in examples:
    #     print(x)
    # x = np.random.randint(0, high=128, size=(4, 2, 8))
    # xx = np.random.randint(high=128, size=(4, 8))

    # y = _tensor_feature(x)
    # print(y)
    # z = tf.train.BytesList(value=[y.numpy()])
    # print('bytes list:', tf.train.BytesList(value=[y.numpy()]))
    # print('tf train feature: ', tf.train.Feature(bytes_list=z))
    # x = tf.io.parse_tensor(y, out_type=tf.dtypes.int32)
    # print(_bytes_feature(b'test_string'))
    # print(_bytes_feature(u'test_bytes'.encode('utf-8')))
