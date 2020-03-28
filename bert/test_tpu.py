import os, re, time, json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)

import numpy as np
import sys
import absl
import bert
import functools
import os


def detect_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver('teodor-cotet', zone=None, project=None) # TPU detection
    except ValueError:
        tpu = None
        
    gpus = tf.config.experimental.list_logical_devices("GPU")


    # Select appropriate distribution strategy
    if tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on single GPU ', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on CPU')
        print("Number of accelerators: ", strategy.num_replicas_in_sync)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)


FLAGS = absl.flags.FLAGS



absl.flags.DEFINE_integer('max_seq_len', 128, 'Maximum sequence length')
# TODO: change default value to None
absl.flags.DEFINE_string('model_folder_path', '../Models/ro0/', 'Path to bert model folder')
absl.flags.DEFINE_float('learning_rate', 1e-5, 'Learning Rate used for optimization')
absl.flags.DEFINE_integer('batch_size', 32, 'Batch size to use during training')
absl.flags.DEFINE_integer('epochs', 1, 'Number of epochs to train')
absl.flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate')
absl.flags.DEFINE_integer('num_classes', 4, "Number of classes for clasification task")
absl.flags.DEFINE_integer('experiment_index', 1, 'Index of current experiment. Will be appended to weights file')
absl.flags.DEFINE_string('save_folder_path',".", "Save folder prefix")
absl.flags.DEFINE_bool("use_tpu", False, "Use TPU or not")
absl.flags.DEFINE_string("tpu_name", None, "Name of TPU instance")


if __name__ == "__main__":
  # create model
  # if FLAGS.use_tpu == True:
  #     tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=None, project=None)
  #     tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
  #     tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
  #     strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
  #     with strategy.scope():
  detect_accelerator()