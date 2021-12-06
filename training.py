import random
import numpy as np
import os
import nibabel
import time
import cv2
import subprocess
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt

print(tf.__version__)

#def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#install('tensorflow==2.4.1')

random.seed(42)

## Load data ##
MaxNumberOfSamples = 200000
width, height = 128, 128
n_channels = 1
bval = 0
n_bvals = 15 #number of b_vals per subject used for training
augmentation = False
local = False

if local:
    dataPath = '/home/hkebiri/Desktop/DHCP_Preterm37_Data'
else:
    dataPath = '/data/bach/Preterm_DHCP-DWI'
    
images = np.zeros((MaxNumberOfSamples, width, height, n_channels))
slice_counter = 0

test_set = ['sub-CC00760XX10', 'sub-CC00764AN14', 'sub-CC00764BN14', 'sub-CC00788XX22'] #F1
#test_set = ['sub-CC00627XX17', 'sub-CC00670XX11', 'sub-CC00703XX10', 'sub-CC00735XX18'] #F2
#test_set = ['sub-CC00492BN15', 'sub-CC00563XX11', 'sub-CC00570XX10', 'sub-CC00571AN11'] #F3
#test_set = ['sub-CC00293BN14', 'sub-CC00351XX05', 'sub-CC00423XX11', 'sub-CC00492AN15'] #F4
#test_set = ['sub-CC00238AN16', 'sub-CC00238BN16', 'sub-CC00281AN10', 'sub-CC00293AN14'] #F5
#test_set = ['sub-CC00177XX13', 'sub-CC00216AN10', 'sub-CC00216BN10', 'sub-CC00231XX09'] #F6 sub-CC00216BN10 is empty
#test_set = ['sub-CC00129BN14', 'sub-CC00132XX09', 'sub-CC00147XX16', 'sub-CC00161XX05'] #F7
#test_set = ['sub-CC00063AN06', 'sub-CC00087BN14', 'sub-CC00124XX09', 'sub-CC00129AN14'] #F8

for path, subdirs, files in os.walk(dataPath):
    for name in files:
         if name.startswith('sub-') and name.endswith("dwi.nii.gz") and not any(elem in path for elem in test_set) and slice_counter < MaxNumberOfSamples:
            img_nib = nibabel.load(os.path.join(path, name)) # Load data
            image_data = img_nib.get_data()
            
            root = name.split(".")[0] 
            bvals = open(os.path.join(path, root+".bval"), "r")
            bvals = bvals.read().split(' ')
            bval_idxes = [i for i, e in enumerate(bvals) if float(e) == bval]
            bval_idxes = bval_idxes[:n_bvals]
            print(root)        
            for i in bval_idxes: 
                for j in range(image_data.shape[2]): 
                    curr_slice = image_data[:,:,j,i]
                    if slice_counter < MaxNumberOfSamples:
                        if np.max(curr_slice) != 0:
                            images[slice_counter,:,:,0] = curr_slice/np.max(curr_slice) #normalize

                        slice_counter += 1 
                    else:
                        break
images_train = images[0:slice_counter,:,:,:]        

print("Shape training data",images_train.shape)
print("Slices: ", slice_counter)
print("Training data size:", slice_counter*width*height, 'voxels')

class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        #self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def call(self, inputs,training=False):
        x = self.encoder(inputs)
        output = self.decoder(x)
        return output

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker
            #self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x = self.encoder(data)
            reconstruction = self.decoder(x)
            mse = keras.losses.MeanSquaredError()
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    #keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    mse(data, reconstruction)
                )
            )
            print("TRAIN STEP")

            total_loss = reconstruction_loss #+ self.alpha*kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
            #"kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):

        x = self.encoder(data)
        reconstruction = self.decoder(x)
        mse = keras.losses.MeanSquaredError()
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                #keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                mse(data,reconstruction)
            )
        )
        print("VAL STEP")

        total_loss = reconstruction_loss 

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss) #

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
            #"kl_loss": self.kl_loss_tracker.result(),
        }


def model_builder(hp):

    # Tuning the latent dimension 
    hp_latent_fm = 32 #hp.Choice('latent_fm', values = [8,16,32,64])
    k_size = 3 #kernel_size 
    network = 'upsample'
    #activity_reg = hp.Choice('activity_reg', values = [0.0, 0.01,0.1])
    #kernel_reg = hp.Choice('kernel_reg', values = [0.01,0.1])

    ## Encoder ##
    encoder_inputs = keras.Input(shape=(width, height, 1))
    if augmentation:
        data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),])
        x = data_augmentation(encoder_inputs)
    else:
        x=encoder_inputs
    #Block 1
    x = layers.Conv2D(32, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(64, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    #Block 2
    x = layers.AvgPool2D(pool_size = (2,2), strides=None,  padding="same")(x)
    x = layers.Conv2D(64, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(128, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    #Block 3
    x = layers.AvgPool2D(pool_size = (2,2), strides=None,  padding="same")(x)
    x = layers.Conv2D(128, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(256, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    #Block 4
    x = layers.AvgPool2D(pool_size = (2,2), strides=None,  padding="same")(x)
    x = layers.Conv2D(256, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(512, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)

    #x = layers.Flatten()(x)
    x = layers.Conv2D(256, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(hp_latent_fm, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)

    x = layers.Flatten()(x)

    encoder = keras.Model(encoder_inputs, x, name="encoder")
    encoder.summary()

    ###### Decoder ######
    latent_inputs = keras.Input(shape=(16*16*hp_latent_fm,))

    x = latent_inputs
    x = layers.Reshape((16,16,hp_latent_fm))(x)
    
    #Block -4
    x = layers.Conv2D(512, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(256, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)

    #Block -3
    if network == 'upsample':
        x = layers.UpSampling2D(size=(2, 2),interpolation="nearest")(x)
    else:
        x = layers.Conv2DTranspose(256, k_size, activation=None, use_bias=False, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(keras.activations.elu)(x)

    x = layers.Conv2D(256, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(128, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)

    #Block -2
    if network == 'upsample':
        x = layers.UpSampling2D(size=(2, 2),interpolation="nearest")(x)
    else:
        x = layers.Conv2DTranspose(128, k_size, activation=None, use_bias=False, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(keras.activations.elu)(x)    

    x = layers.Conv2D(128, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(64, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)

    #Block -1
    if network == 'upsample':
        x = layers.UpSampling2D(size=(2, 2),interpolation="nearest")(x)
    else:    
        x = layers.Conv2DTranspose(64, k_size, activation=None, use_bias=False, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(keras.activations.elu)(x)

    x = layers.Conv2D(64, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)
    x = layers.Conv2D(32, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)

    x = layers.Conv2D(32, k_size, activation=None, use_bias=False, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.elu)(x)

    decoder_outputs = layers.Conv2D(1, k_size, activation="sigmoid", padding="same")(x) 
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    # Tuning the learning rate for the optimizer 
    hp_learning_rate = 5e-5 #hp.Choice('learning_rate', values = [1e-4, 5e-4, 1e-5, 5e-5, 1e-6]) 

    model = AE(encoder,decoder)

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate))

    return model


## Train it ##

# #Saving weights params
checkpoint_path = "/data/bach/AE/in-vivo-DHCP/hyperparams_Bayesian_tuningB0ValNoKL_MSE_MoreEpochs_Fold1/cp-{epoch:03d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=689*30) #i.e. save every 30 epochs because #36*3 (810 batches = 1 epoch)

#Nan cases
nan = tf.keras.callbacks.TerminateOnNaN()

#Tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(
log_dir='/data/bach/AE/in-vivo-DHCP/logs_hyperparamsB0ValNoKL_MAE_MoreEpochs_Fold1', histogram_freq=0, write_graph=False,
write_images=False, update_freq='epoch', profile_batch=2,
embeddings_freq=0, embeddings_metadata=None)


#Start training
start = time.time()

training_data = images_train.astype("float32")

##Keras-tuner
tuner = kt.BayesianOptimization(model_builder,
                     objective = kt.Objective("val_total_loss", direction="min"), 
                     max_trials = 2, 
                     project_name = 'hyperparams_Bayesian_tuningB0ValNoKL_MSE_MoreEpochs_Fold1',
                     directory='/data/bach/AE/in-vivo-DHCP')


tuner.search(training_data, validation_split = 0.15, batch_size = 32, epochs = 200, callbacks = [cp, tensorboard, nan])
print("Training took", (time.time() - start)/60, "min")

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete and the best configuration is {best_hps}
""")
