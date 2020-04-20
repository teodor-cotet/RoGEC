import tensorflow as tf
from transformer.bert_encoder_layer import BertEncoder
from transformer.decoder import Decoder

class TransformerBert(tf.keras.Model):

    def __init__(self, num_layers=None, d_model=None, num_heads=None, dff=None,
                input_vocab_size=None, 
                target_vocab_size=None, model_dir=None, pe_input=None, pe_target=None, rate=0.1, 
                decoder=None, final_layer=None, args=None):
        super(TransformerBert, self).__init__()

        self.encoder = BertEncoder(model_dir=model_dir, d_model=d_model, args=args)
        if decoder:
            self.decoder = decoder
        else:
            self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                            target_vocab_size, pe_target, rate)
        if final_layer:
            self.final_layer = final_layer
        else:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, input_ids, input_seg, tar, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_ids, input_seg, training)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights