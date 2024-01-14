class Config:
    DATA_PATH = '../../data/setup_1.pkl'
    RANDOM_SEED = 0
    VALIDATE_SIZE = 0.2
    BATCH_SIZE = 256 # 128 / 256
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MODEL_TYPE = 'Autoencoder'  # Options: 'Autoencoder', 'DenoisingAE'
    NUM_LAYERS = 6  # Total number of layers in Autoencoder (must be even number!) 4 / 6 / 8
    BOTTLENECK_SIZE = 4 # 2 4 6 8 10
    ACTIVATION_FUNC = 'ELU'  # Options: 'ReLU', 'ELU', 'Tanh'
    PATIENCE = 10
    MIN_DELTA = 0.001 # 0.01 / 0.001
    LOSS_FUNCTION = 'MSE'  # Options: 'BCE', 'MSE'
    PLOT_CLIPPING_VALUE = 6 # Good Options: 0.05 (BCE) or 10 (MSE)
    NOISE_FACTOR = 0.5 # For DenoisingAE

"""
autoencoders:

batch size: 2
model types: 1
num layers: 3
bottleneck_size: 5
activation: 3

OCNN:

batch size: 2
nu: 5 (0.0001, 0.0005, 0.001, 0.005, 0.01)
r: 3 (0.1, 1, 10)

"""