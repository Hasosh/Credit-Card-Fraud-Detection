class Config:
    # RELEVANT FOR ALL MODELS
    DATA_PATH = '../../data/setup_1.pkl'
    RANDOM_SEED = 0
    VALIDATE_SIZE = 0.2
    BATCH_SIZE = 256 # 128 / 256
    EPOCHS = 30
    LEARNING_RATE = 1e-4 # 1e-2 or 1e-3 for autoencoder, 1e-4 for OCNN
    PATIENCE = 5 # 10 for autoencoder, less for OCNN
    MIN_DELTA = 0.001 # 0.01 / 0.001 (AE / OCNN)
    USE_WANDB = True

    # RELEVANT ONLY FOR AUTOENCODER
    MODEL_TYPE = 'CustomVAE'  # Options: 'Autoencoder', 'DenoisingAE', 'CustomAE', 'CustomDAE', 'CustomVAE'
    NUM_LAYERS = 4  # Total number of layers in Autoencoder (must be even number!) 4 / 6 / 8
    BOTTLENECK_SIZE = 1 # 2 4 6 8 10
    ACTIVATION_FUNC = 'ELU'  # Options: 'ReLU', 'ELU', 'Tanh'
    LOSS_FUNCTION = 'MSE'  # Options: 'BCE', 'MSE'
    PLOT_CLIPPING_VALUE = 6 # Good Options: 0.05 (BCE) or 10 (MSE)
    NOISE_FACTOR = 0.5 # For DenoisingAE

    # RELEVANT ONLY FOR OCNN 
    PARAMETER_NU = 0.001
    PARAMETER_R = 1.0
    HIDDEN_LAYER_SIZE = 16

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