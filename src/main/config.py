# config.py
class Config:
    DATA_PATH = '../../data/setup_1.pkl'
    RANDOM_SEED = 0
    VALIDATE_SIZE = 0.2
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MODEL_TYPE = 'DenoisingAE'  # Options: 'Autoencoder', 'DenoisingAE'
    NUM_LAYERS = 4  # Total number of layers in Autoencoder (must be even number!)
    BOTTLENECK_SIZE = 4
    ACTIVATION_FUNC = 'ELU'  # Options: 'ReLU', 'ELU', 'Tanh'
    PATIENCE = 10
    MIN_DELTA = 0.01
    LOSS_FUNCTION = 'MSE'  # Options: 'BCE', 'MSE'
    PLOT_CLIPPING_VALUE = 10 # Good Options: 0.05 (BCE) or 10 (MSE)
    NOISE_FACTOR = 0.5 # For DenoisingAE
