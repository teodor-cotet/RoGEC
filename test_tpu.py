import tensorflow as tf
import numpy as np
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

def gen_test():
    data = np.random.uniform(.0, 1.0, (1024, 8))
    labels = np.random.randint(2, size=(1024,))

    for i, _ in enumerate(data):
        yield data[i], labels[i]

def run_model(strategy):
    batch_size = 64
    # dataset = tf.data.Dataset.from_generator(
    #         generator=gen_test, 
    #         output_types=(tf.int32, tf.int32),
    #         output_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
    it = gen_test()
    samples = list(it)
    samples_data = [s[0] for s in samples]
    samples_labels = [s[1] for s in samples]

    dataset = tf.data.Dataset.from_tensor_slices((samples_data, samples_labels))
    dataset = dataset.cache().repeat().batch(batch_size, drop_remainder=True)

    if args.use_tpu:
        with strategy.scope():
            inp = tf.keras.Input(shape=(8,))
            x = tf.keras.layers.Dense(4, activation='relu')(inp)
            y = tf.keras.layers.Dense(2, activation='softmax')(x)
            model = tf.keras.Model(inputs=inp, outputs=y)
            optimizer = tf.keras.optimizers.SGD()
            model.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['sparse_categorical_accuracy'])
            model.fit(dataset, epochs=100, steps_per_epoch=1024//batch_size)
    else:
        inp = tf.keras.Input(shape=(8,))
        x = tf.keras.layers.Dense(4, activation='relu')(inp)
        y = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.Model(inputs=inp, outputs=y)
        optimizer = tf.keras.optimizers.SGD()
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy'])
        model.fit(dataset, epochs=100, steps_per_epoch=1024//batch_size)

def main(argv):
    del argv

    if args.use_tpu:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu,
                zone=args.tpu_zone, project=args.gcp_project)
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
        print('Running on TPU ', tpu_cluster_resolver.cluster_spec().as_dict()['worker'])
        run_model(strategy)
    else:
        run_model(None)
if __name__ == "__main__":
    absl_app.run(main)