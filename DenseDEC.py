from keras.layers import Dense, Input, Flatten, Reshape, BatchNormalization, Activation, Dropout
from keras.models import Sequential, Model
from keras.initializers import VarianceScaling

from DEC import DEC

def DenseAutoencoder(input_dim, encoding_dim=64):
    """
    Creates a dense autoencoder with a four-layer encoder for large binary input.
    The encoder reduces dimensionality smoothly from the input dimension to the encoding dimension,
    and the decoder symmetrically reconstructs back to the input dimension.

    Args:
    input_dim (int): The input dimension, expected to be large (around 40000)
    encoding_dim (int): The encoding dimension, default is 64

    Returns:
    tuple: (autoencoder, encoder) - The full autoencoder model and the encoder part
    """
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    model = Sequential()

    # Determine intermediate dimensions for a smoother transition
    inter_dim1 = min(max(encoding_dim * 32, input_dim // 4), input_dim // 2)
    inter_dim2 = min(max(encoding_dim * 16, input_dim // 8), inter_dim1 // 2)
    inter_dim3 = min(max(encoding_dim * 8, input_dim // 16), inter_dim2 // 2)

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
    encoder = Model(inputs=model.input, outputs=model.get_layer(index=11).output)

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
