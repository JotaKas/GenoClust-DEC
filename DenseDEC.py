from keras.layers import Dense, Input, Flatten, Reshape
from keras.models import Sequential, Model
from keras.initializers import VarianceScaling
# Removed the import of ImageDataGenerator and Conv2D layers since they are not used for genomic data

from DEC import DEC

def DenseAutoencoder(input_dim, encoding_dim=64):
    """
    Creates a dense autoencoder architecture suitable for genomic data.
    """
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    model = Sequential()
    # Encoder
    model.add(Dense(encoding_dim, activation='relu', input_dim=input_dim, kernel_initializer=init, name='encoder'))
    # Decoder
    model.add(Dense(input_dim, activation='sigmoid', kernel_initializer=init, name='decoder'))

    # Extract the encoder part of the model
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)

    return model, encoder

class DenseDEC(DEC):
    def __init__(self, input_dim, encoding_dim=64, init='glorot_uniform'):
        """
        Initializes the DEC model adapted for genomic data using a dense autoencoder.
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder, self.encoder = DenseAutoencoder(input_dim, encoding_dim)
