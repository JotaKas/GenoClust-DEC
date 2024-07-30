from keras.layers import Dense, Input, Flatten, Reshape, BatchNormalization, Activation, Dropout
from keras.models import Sequential, Model
from keras.initializers import VarianceScaling

from DEC import DEC

def DenseAutoencoder(input_dim, encoding_dim=64):
    """
    Creates a dense autoencoder with a four-layer encoder. The encoder reduces dimensionality smoothly
    from the input dimension to the encoding dimension, and the decoder symmetrically reconstructs back to the input dimension.
    """
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    model = Sequential()

    # Determine intermediate dimensions for a smoother transition
    inter_dim1 = max(encoding_dim * 4, int(input_dim / 2))
    inter_dim2 = max(encoding_dim * 2, int(inter_dim1 / 2))
    inter_dim3 = encoding_dim * 2  # Closer to the latent space but higher than the final encoding dimension

    # Encoder
    model.add(Dense(inter_dim1, input_dim=input_dim, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(inter_dim2, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(inter_dim3, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(encoding_dim, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Decoder
    model.add(Dense(inter_dim3, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(inter_dim2, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(inter_dim1, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(input_dim, activation='sigmoid', kernel_initializer=init))

    # Extract the encoder part of the model
    encoder = Model(inputs=model.input, outputs=model.layers[7].output)

    return model, encoder

class DenseDEC(DEC):
    def __init__(self, input_dim, encoding_dim=64, init='glorot_uniform'):
        """
        Initializes the DEC model adapted for genomic data using a dense autoencoder.
        """
        dims = [input_dim, encoding_dim]
        super().__init__(dims, init)
        self.encoding_dim = encoding_dim
        self.autoencoder, self.encoder = DenseAutoencoder(input_dim, encoding_dim)
