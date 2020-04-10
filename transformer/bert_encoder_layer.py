import tensorflow as tf
import bert

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, model_dir, d_model):
        super(BertEncoder, self).__init__()
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")
        tf.compat.v1.logging.info('bert model loaded from {}'.format(model_dir))
            
    def call(self, input_ids, segment_ids, training):
        bert_output = self.bert_layer([input_ids, segment_ids])
        return bert_output  # (batch_size, input_seq_len, d_model)