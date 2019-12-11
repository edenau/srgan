import numpy as np
import os
import tensorflow as tf
from nets import ReconNet

def main(training, train_data_filename=None, val_data_filename=None, load_ckpt_dir=None, save_ckpt_dir=None):
    # 0) Initialise
    src_dir = os.path.dirname(os.path.abspath(__file__))
    model_pdir = os.path.join(src_dir, '..', 'models')

    print(tf.config.experimental.list_physical_devices())
    strategy = tf.distribute.MirroredStrategy()#cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        # 1) Compile model
        final_bias = 0.3287
        model = ReconNet(1, final_bias=final_bias)

        print(f'#in sync: {strategy.num_replicas_in_sync}')
        batch_size = 16 * strategy.num_replicas_in_sync
        #loss_func = tf.keras.losses.MSE
        lr_schedule = 1e-4
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer,loss='mse')

    # 2) Load weights (if requested)
    if load_ckpt_dir is not None:
        ckpt_dir = os.path.join(model_pdir, load_ckpt_dir)
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        model.load_weights(latest_ckpt)
        print(f'Model loaded from {load_ckpt_dir}.')

    if training:
        # 3) Load training data
        data_pdir = os.path.join(src_dir, '..', 'data')

        train_data_path = os.path.join(data_pdir, train_data_filename)
        train_data = np.load(train_data_path) # Assume data are normalised to [-1,1]
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data)).batch(batch_size)
        print(f'Training data {train_data_filename} loaded.')

        if val_data_filename is None:
            val_dataset = None
        else:
            # 3a) Load validation data
            val_data_path = os.path.join(data_pdir, val_data_filename)
            val_data = np.load(val_data_path) # Assume data are normalised to [-1,1]
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_data)).batch(batch_size)
            print(f'Validation data {val_data_filename} loaded.')

        # 4) Train model
        ckpt_path = os.path.join(model_pdir, save_ckpt_dir, 'weights-{epoch:02d}.ckpt')
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                           save_weights_only=True,
                                                           verbose=1)
        model.fit(train_dataset,
                  epochs=20,
                  callbacks=[ckpt_callback],
                  validation_data=val_dataset)

        final_ckpt_path = os.path.join(model_pdir, save_ckpt_dir, 'weights-final.ckpt')
        model.save_weights(final_ckpt_path)
    return model


if __name__ == '__main__':

    main(training=True,
         train_data_filename='train.npy',
         val_data_filename=None,
         load_ckpt_dir=None,
         save_ckpt_dir='02')
