# config.py
class Config:
    DATA_PATH = '../../data/setup_1.pkl'
    RANDOM_SEED = 0
    VALIDATE_SIZE = 0.2
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MODEL_TYPE = 'Autoencoder'  # Options: 'Autoencoder', 'VariationalAutoencoder'
    NUM_LAYERS = 6  # Total number of layers in Autoencoder
    BOTTLENECK_SIZE = 4
    ACTIVATION_FUNC = 'ELU'  # Options: 'ReLU', 'ELU', 'Tanh'
    PATIENCE = 10
    MIN_DELTA = 0.001
    LOSS_FUNCTION = 'MSE'  # Options: 'BCE', 'MSE'
    PLOT_CLIPPING_VALUE = 10 # Good Options: 0.05 (BCE) or 10 (MSE)
