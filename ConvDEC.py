from keras.layers import Dense, Input, Flatten, Reshape
from keras.models import Sequential, Model
from keras.initializers import VarianceScaling

from DEC import DEC

def DAE(input_shape=(784,), units=[512, 256, 128]):
    """
    Dense Autoencoder for genomic data.
    
    Arguments:
    input_shape -- The shape of the input data. For one-hot encoded genomic data,
                   this would typically be (number_of_genes,).
    units -- List of units for each dense layer in the encoder. The decoder is symmetric.
    
    Returns:
    model -- The autoencoder model.
    encoder -- The encoder part of the model.
    """
    model = Sequential()
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    
    # Encoder
    model.add(Input(shape=input_shape))
    for unit in units:
        model.add(Dense(unit, activation='relu', kernel_initializer=init))
    
    # Embedding
    model.add(Dense(units[-1], name='embedding', kernel_initializer=init))  # bottleneck layer
    
    # Decoder
    for unit in reversed(units[:-1]):
        model.add(Dense(unit, activation='relu', kernel_initializer=init))
    
    model.add(Dense(input_shape[0], activation='sigmoid', kernel_initializer=init))
    
    encoder = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
    
    return model, encoder

class ConvDEC(DEC):
    def __init__(self, input_shape, units=[512, 256, 128], init='glorot_uniform'):
        """
        Initializes the ConvDEC model adapted for genomic data.
        
        Arguments:
        input_shape -- Shape of the input data.
        units -- Architecture of the dense autoencoder.
        init -- Weight initialization method.
        """
        self.input_shape = input_shape
        self.autoencoder, self.encoder = DAE(input_shape, units)
        super(ConvDEC, self).__init__(input_shape=input_shape, init=init)
