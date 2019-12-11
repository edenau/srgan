import numpy as np
import os
import tensorflow as tf
import time
from nets import ReconNet, Discriminator

def main(training, train_data_filename=None, pretrained_G_dir=None, save_ckpt_dir=None):
    # 0) Initialise
    src_dir = os.path.dirname(os.path.abspath(__file__))
    model_pdir = os.path.join(src_dir, '..', 'models')

    print(tf.config.experimental.list_physical_devices())
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        print(f'#in sync: {strategy.num_replicas_in_sync}')
        batch_size = 1# * strategy.num_replicas_in_sync

        # 1) Get model
        G = ReconNet(1)
        ckpt_dir = os.path.join(model_pdir, pretrained_G_dir)
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        G.load_weights(latest_ckpt)
        print(f'Generator loaded from {pretrained_G_dir}.')
        #G_lr_schedule = 1e-4
        G_lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[1e5], values=[1e-4, 1e-5])
        G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_schedule)

        D = Discriminator()
        #D_lr_schedule = 1e-4
        D_lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[1e5], values=[1e-4, 1e-5])
        D_optimizer = tf.keras.optimizers.Adam(learning_rate=D_lr_schedule)

    if training:
        # 3) Load training data
        data_pdir = os.path.join(src_dir, '..', 'data')

        train_data_path = os.path.join(data_pdir, train_data_filename)
        train_data = np.load(train_data_path) # Assume data are normalised to [-1,1]
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data)).batch(batch_size)
        print(f'Training data {train_data_filename} loaded.')

        # 4) Train model
        epochs = 10
        ckpt_path = os.path.join(model_pdir, save_ckpt_dir, 'ckpt')
        train([G,D], [G_optimizer, D_optimizer], train_dataset, epochs, ckpt_path)
    return G, D


@tf.function
def train_step(models, optimizers, batch):
    def get_D_loss(real_prob, fake_prob):
        cross_entropy = tf.keras.losses.BinaryCrossentropy()
        real_loss = cross_entropy(tf.ones_like(real_prob), real_prob)
        fake_loss = cross_entropy(tf.zeros_like(fake_prob), fake_prob)
        return real_loss + fake_loss

    def get_G_loss(real, fake, fake_prob, xent2mse=1e-3):
        cross_entropy = tf.keras.losses.BinaryCrossentropy()
        return tf.keras.losses.MSE(real, fake) + xent2mse * cross_entropy(tf.ones_like(fake_prob), fake_prob)

    G, D = models
    G_optimizer, D_optimizer = optimizers
    input, real = batch

    with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
        fake = G(input, training=True)
        real_prob = D(real, training=True)
        fake_prob = D(fake, training=True)

        G_loss = get_G_loss(real, fake, fake_prob)
        D_loss = get_D_loss(real_prob, fake_prob)
        G_grad = G_tape.gradient(G_loss, G.trainable_variables)
        D_grad = D_tape.gradient(D_loss, D.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))
    return G_loss, D_loss


def train(models, optimizers, dataset, epochs, ckpt_path):
    G, D = models
    G_optimizer, D_optimizer = optimizers
    checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer, D_optimizer=D_optimizer,
                                     G=G, D=D)

    log_G_loss = tf.keras.metrics.Mean('avg_G_loss', dtype=tf.float32)
    log_D_loss = tf.keras.metrics.Mean('avg_D_loss', dtype=tf.float32)

    for epoch in range(epochs):
        start = time.time()

        for batch_id, batch in enumerate(dataset):
            G_loss, D_loss = train_step(models, optimizers, batch)
            log_G_loss.update_state(G_loss)
            log_D_loss.update_state(D_loss)
            print(f'{epoch} - {batch_id} - G loss: {log_G_loss.result():.4f} - D loss: {log_D_loss.result():.4f}')
        checkpoint.save(file_prefix=ckpt_path)
        print (f'Time for epoch {epoch+1}: {time.time()-start} sec')
        log_G_loss.reset_states()
        log_D_loss.reset_states()

if __name__ == '__main__':

    main(training=True,
         train_data_filename='train.npy',
         pretrained_G_dir=None,
         save_ckpt_dir='gan-01')
