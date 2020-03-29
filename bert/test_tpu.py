from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# pylint: disable=g-bad-import-order
from absl import app as absl_app  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf
# pylint: enable=g-bad-import-order

# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default='teodor-cotet',
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default='us-central1-f',
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default='rogec-271608',
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS
print(FLAGS.gcp_project)

def main(argv):
    del argv  # Unused.
    tf.logging.set_verbosity(tf.logging.INFO)

    # tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    #     FLAGS.tpu,
    #     zone=FLAGS.tpu_zone,
    #     project=FLAGS.gcp_project
    # )
    # # added by me
    # if tpu_cluster_resolver:
    #     tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    #     strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver, steps_per_run=128) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    #     print('Running on TPU ', tpu_cluster_resolver.cluster_spec().as_dict()['worker'])  
    # else:
    #     print('no tpu')

    if FLAGS.use_tpu == True:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver, steps_per_run=128)
        print('Running on TPU ', tpu_cluster_resolver.cluster_spec().as_dict()['worker'])
    else:
        print(FLAGS.use_tpu)
        # print(FLAGS)
        # for k in FLAGS.__wrapped.__dict__:
        #     if FLAGS.__dict__[k] is not None:
        #         print(k, '->', FLAGS.__dict__[k])
    #     with strategy.scope():
    #         model, bert = create_model()
    # else:
    #         model, bert = create_model()

    # run_config = tf.estimator.tpu.RunConfig(
    #     cluster=tpu_cluster_resolver,
    #     model_dir=FLAGS.model_dir,
    #     session_config=tf.ConfigProto(
    #         allow_soft_placement=True, log_device_placement=True),
    #     tpu_config=tf.estimator.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
    # )

    # estimator = tf.estimator.tpu.TPUEstimator(
    #     model_fn=model_fn,
    #     use_tpu=FLAGS.use_tpu,
    #     train_batch_size=FLAGS.batch_size,
    #     eval_batch_size=FLAGS.batch_size,
    #     predict_batch_size=FLAGS.batch_size,
    #     params={"data_dir": FLAGS.data_dir},
    #     config=run_config)
    # TPUEstimator.train *requires* a max_steps argument.
    # estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
    # # TPUEstimator.evaluate *requires* a steps argument.
    # # Note that the number of examples used during evaluation is
    # # --eval_steps * --batch_size.
    # # So if you change --batch_size then change --eval_steps too.
    # if FLAGS.eval_steps:
    #     estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)

    # # Run prediction on top few samples of test data.
    # if FLAGS.enable_predict:
    #     predictions = estimator.predict(input_fn=predict_input_fn)

    # for pred_dict in predictions:
    #     template = ('Prediction is "{}" ({:.1f}%).')

    #     class_id = pred_dict['class_ids']
    #     probability = pred_dict['probabilities'][class_id]

    #     print(template.format(class_id, 100 * probability))
if __name__ == "__main__":
    # tf.disable_v2_behavior()
    absl_app.run(main)