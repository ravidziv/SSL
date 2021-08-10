"""SimSLr tf.keras implementation
    Chen, Ting, et al. "A simple framework for contrastive learning of visual representations. ICML, 2020."""
import tensorflow as tf
from self_supervised.loss import cont_loss2
from self_supervised.models.model import SupervisedHead, ProjectionHead
from self_supervised.models import resent_model


class SimCLR(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model = None, projection_head: tf.keras.Model = None,
                 num_classes: int = 10, projection_head_args=None, resent_head_args=None,
                 train_mode: str = None, lineareval_while_pretraining: bool = True):
        """
        SimCLR tf model 
        :rtype: None
        :param encoder: The encoder for the model
        :param projection_head: the projection head model
        :param num_classes: number of classes
        :param projection_head_args:  the arguments for the projection head
        :param resent_head_args:  the arguments for the encoder 
        :param train_mode: What is the train mode - pretraining or fine-tuning 
        :param lineareval_while_pretraining: If we want to train the supervised head on the top
        """
        
        super(SimCLR, self).__init__()
        if encoder is None:
            self.encoder = resent_model.resnet(**resent_head_args)
        else:
            self.encoder = encoder
        if projection_head is None:
            self.projection_head = ProjectionHead(**projection_head_args)
        else:
            self.projection_head = projection_head
        self.num_classes = num_classes
        self.supervised_head = SupervisedHead(self.num_classes)
        self.train_mode = train_mode
        self.lineareval_while_pretraining = lineareval_while_pretraining
        self.contrastive_optimizer = None
        self.probe_optimizer = None
        self.probe_loss = None
        self.contrastive_loss_tracker = None
        self.contrastive_accuracy = None
        self.probe_loss_tracker = None
        self.probe_accuracy = None
        self.contrastive_loss = None

    def compile(self, contrastive_optimizer, probe_optimizer, temperature, **kwargs):
        """
        :param contrastive_optimizer:  The optimizer for the contrastive learning
        :param probe_optimizer:  the optimizer for the supervised head
        :param temperature: the temperature of the loss
        :param kwargs:
        """
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer
        # self.contrastive_loss will be defined as a method
        self.probe_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.SUM)
        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = tf.keras.metrics.CategoricalAccuracy(name="c_acc")
        self.probe_loss_tracker = tf.keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = tf.keras.metrics.CategoricalAccuracy(name="p_acc")
        self.contrastive_loss = cont_loss2(temperature=temperature)

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def get_config(self):
        return {"encoder": self.encoder, "projection_head": self.decoder}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None, mask=None):
        features = inputs
        # if training and self.train_mode == 'pretrain':
        #    #if self.fine_tune_after_block > -1:
        #    #    raise ValueError('Does not support layer freezing during pretraining,'
        #    #                     'should set fine_tune_after_block<=-1 for safety.')
        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')
        num_transforms = inputs.shape[3] // 3
        num_transforms = tf.repeat(3, num_transforms)
        # Split channels, and optionally apply extra batched augmentation.
        features_list = tf.split(
            features, num_or_size_splits=num_transforms, axis=-1)
        # if self.use_blur and training and self.train_mode == 'pretrain':
        #    features_list = data_utils.batch_random_blur(features_list,
        #                                                self.image_size,
        #                                                self.image_size)
        features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)
        # Base network forward pass.
        hidden = self.encoder(features, training=training)
        # Add heads.
        projection_head_outputs, supervised_head_inputs = self.projection_head(
            hidden, training)

        if self.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs
        elif self.train_mode == 'pretrain' and self.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)
            return projection_head_outputs, supervised_head_outputs
        else:
            return projection_head_outputs, None

    def train_step(self, data):
        # Forward pass through the encoder and predictor.
        features, labels = data[0], data[1]
        # images = tf.concat((unlabeled_images, labeled_images), axis=0)
        with tf.GradientTape(persistent=True) as tape:
            projection_head_outputs, supervised_head_outputs = self(features, training=True)
            if projection_head_outputs is not None:
                outputs = projection_head_outputs
                con_loss, logits_con, labels_con = self.contrastive_loss(outputs)
            con_loss += sum(self.losses)
            # Labels are only used in evaluation for an on-the-fly logistic regression
            if supervised_head_outputs is not None:
                outputs = supervised_head_outputs
                new_labels = labels
                # class_logits = self.linear_probe(tf.stop_gradient(features))
                if self.train_mode == 'pretrain' and self.lineareval_while_pretraining:
                    new_labels = tf.concat([new_labels, new_labels], 0)
                probe_loss = self.probe_loss(new_labels, outputs)
        gradients = tape.gradient(probe_loss, self.supervised_head.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.supervised_head.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(new_labels, outputs)
        # Compute gradients and update the parameters.
        gradients = tape.gradient(
            con_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(con_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Forward pass through the encoder and predictor.
        features, labels = data
        # For testing the components are used with a training=False flag
        _, supervised_head_outputs = self(features, training=False)
        new_labels = labels
        if new_labels.shape[0] < supervised_head_outputs.shape[0]:
            new_labels = tf.concat([new_labels, new_labels], 0)
        probe_loss = self.probe_loss(new_labels, supervised_head_outputs)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(new_labels, supervised_head_outputs)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
