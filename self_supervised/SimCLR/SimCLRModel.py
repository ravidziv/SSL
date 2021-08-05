"""SimSLr tf.keras implementation
    Chen, Ting, et al. "A simple framework for contrastive learning of visual representations. ICML. PMLR, 2020."""

import tensorflow as tf


class SimCLR(tf.keras.Model):

    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model = None,
                 finetune_decoder: tf.keras.Model = None):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.combined_model = tf.keras.Sequential([encoder, decoder])
        # self.finetune_mode = False
        # self.finetune_decoder = finetune_decoder
        # self.combined_model = combined_model

    def get_config(self):
        return {"encoder": self.encoder, "decoder": self.decoder}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None, mask=None):
        ds_one, ds_two = inputs['x'], inputs['x2']
        z1, z2 = self.combined_model(ds_one), self.combined_model(ds_two)
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        return z1, z2

    def train_step(self, data):
        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self(data, training=True)
            loss = self.compiled_loss(z1, z2, regularization_losses=self.losses)
        # Compute gradients and update the parameters.
        trainable_vars = self.combined_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Forward pass through the encoder and predictor.
        z1, z2 = self(data, training=True)
        loss = self.compiled_loss(z1, z2, regularization_losses=self.losses)
        self.compiled_metrics.update_state(z1, z2)
        return {m.name: m.result() for m in self.metrics}

    def set_finetune_mode(self, decoder: tf.keras.Model = None):
        if decoder:
            self.finetune_decoder = decoder
        self.finetune_mode = True
