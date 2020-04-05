import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
from absl import app as absl_app

tf.compat.v1.flags.DEFINE_bool("use_tpu", False, "Use TPUs rather than plain CPUs")
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
    
args = tf.compat.v1.flags.FLAGS

def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)])

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

def get_dataset_simple(batch_size=32):
    data = np.random.uniform(.0, 1.0, (1024, 28, 28, 1))
    labels = np.random.randint(10, size=(1024,))

    train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((data[:64], labels[:64]))

    train_dataset = train_dataset.repeat(5).batch(batch_size).cache()
    test_dataset = test_dataset.repeat(5).batch(batch_size).cache()
    return train_dataset, test_dataset

def main(argv):
    del argv
    batch_size = 32
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

            train_dataset, test_dataset = get_dataset_simple()

            model.fit(train_dataset, epochs=5, steps_per_epoch=1024//batch_size)
    else:
        model = create_model()
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['sparse_categorical_accuracy'])

        train_dataset, test_dataset = get_dataset_simple()

        model.fit(train_dataset, epochs=5, steps_per_epoch=1024//batch_size)

if __name__ == "__main__":
    absl_app.run(main)