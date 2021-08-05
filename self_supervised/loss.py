import tensorflow as tf
from functools import partial


def nt_xent_func(temperature: float):
    func = partial(nt_xent, temperature=temperature)
    func.__name__ = 'nt_xent_func'
    return func


def nt_xent(z1, z2, temperature):
    """Implements normalized temperature-scaled cross-entropy loss

    Args:
        z1: normalized latent representation of first set of augmented images [N, D]
        z2: normalized latent representation of second set of augmented images [N, D]
        temperature: temperature for softmax. set in config
    Returns:
        loss: contrastive loss averaged over batch (2*N samples)
    """

    # reshape so that the order is z1_1,z2_1,z1_2,z2_2,z1_3,z2_3
    batch_size = z1.shape[0]
    z = tf.concat([z1, z2], axis=0)
    z_ = tf.reshape(tf.transpose(tf.reshape(z, [2, batch_size, -1]), [1, 0, 2]), [batch_size * 2, -1])

    # compute cosine similarity
    # a has order [z1_1*batch_size*2, z1_2*batch_size*2, ...]
    # b has order [z1_1, z1_2, z3_1 ...]
    a = tf.reshape(tf.transpose(tf.tile(tf.reshape(z_, [1, batch_size * 2, -1]), [batch_size * 2, 1, 1]), [1, 0, 2]),
                   [batch_size * 2 * batch_size * 2, -1])
    b = tf.tile(z_, [batch_size * 2, 1])
    sim = cosine_similarity(a, b)
    sim = tf.expand_dims(sim, axis=1) / temperature
    sim = tf.reshape(sim, [batch_size * 2, batch_size * 2])
    sim = tf.math.exp(sim - tf.reduce_max(sim))

    pos_indices = tf.concat([tf.range(1, (2 * batch_size) ** 2, (batch_size * 4) + 2),
                             tf.range(batch_size * 2, (2 * batch_size) ** 2, (batch_size * 4) + 2)], axis=0)
    pos_indices = tf.expand_dims(pos_indices, axis=1)
    pos_mask = tf.zeros(((2 * batch_size) ** 2, 1), dtype=tf.int32)
    pos_mask = tf.tensor_scatter_nd_add(pos_mask, pos_indices, tf.ones((batch_size * 2, 1), dtype=tf.int32))
    pos_mask = tf.reshape(pos_mask, [batch_size * 2, batch_size * 2])
    neg_mask = tf.ones((batch_size * 2, batch_size * 2), dtype=tf.int32) - tf.eye(batch_size * 2, dtype=tf.int32)

    # similarity between z11-z12, z12-z11, z21-22, z22-z21 etc. 
    pos_sim = tf.reduce_sum(sim * tf.cast(pos_mask, tf.float32), axis=1)

    # negative similarity consists of all similarities except i=j
    neg_sim = tf.reduce_sum(sim * tf.cast(neg_mask, tf.float32), axis=1)
    loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(pos_sim / neg_sim, 1e-10, 1.0)))

    return loss


def cosine_similarity(a, b):
    """Computes the cosine similarity between vectors a and b"""
    numerator = tf.reduce_sum(tf.multiply(a, b), axis=1)
    denominator = tf.multiply(tf.norm(a, axis=1), tf.norm(b, axis=1))
    cos_similarity = numerator / denominator
    return cos_similarity
