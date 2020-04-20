import tensorflow as tf
import bert
import os

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, model_dir, d_model, args):
        super(BertEncoder, self).__init__(trainable=False)
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")
        self.model_dir = model_dir
        tf.compat.v1.logging.info('bert model loaded from {}'.format(model_dir))
        tf.compat.v1.logging.info('bert model params: {}'.format(bert_params))
        # do dummy call to build the model indirectly 
        self.bert_layer([tf.zeros([args.batch_size, args.seq_length], dtype=tf.dtypes.int64),
             tf.zeros([args.batch_size, args.seq_length], dtype=tf.dtypes.int64)])
        bert.load_bert_weights(self.bert_layer, os.path.join(self.model_dir, "bert_model.ckpt"))
        tf.compat.v1.logging.info('bert weights loaded')
        
    def call(self, input_ids, segment_ids, training):
        bert_output = self.bert_layer([input_ids, segment_ids])
        
        return bert_output  # (batch_size, input_seq_len, d_model)