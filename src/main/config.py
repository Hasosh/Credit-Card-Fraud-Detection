class Config:
    # RELEVANT FOR ALL MODELS
    DATA_PATH = 'data/setup_1.pkl' # '../../data/setup_1.pkl' / '../../data/setup_1_latent.pkl'
    RANDOM_SEED = 0
    VALIDATE_SIZE = 0.1 # 0.1 / 0.2
    BATCH_SIZE = 256 # 128 / 256
    EPOCHS = 200 # 100 / 200
    LEARNING_RATE = 1e-4 # 1e-2 or 1e-3 for autoencoder, 1e-4 for OCNN
    PATIENCE = 20 # 10 for autoencoder, less for OCNN
    MIN_DELTA = 0.001 # 0.01 / 0.001 (AE / OCNN)
    USE_WANDB = True

    # RELEVANT ONLY FOR AUTOENCODER
    MODEL_TYPE = 'CustomAE'  # Options: 'Autoencoder', 'DenoisingAE', 'CustomAE', 'CustomDAE', 'CustomVAE'
    NUM_LAYERS = 4  # Total number of layers in Autoencoder (must be even number!) 4 / 6 / 8
    BOTTLENECK_SIZE = 8 # 2 4 6 8 10
    ACTIVATION_FUNC = 'ELU'  # Options: 'ReLU', 'ELU', 'Tanh'
    LOSS_FUNCTION = 'MSE'  # Options: 'BCE', 'MSE'
    PLOT_CLIPPING_VALUE = 6 # Good Options: 0.05 (BCE) or 10 (MSE)
    NOISE_FACTOR = 0.5 # only for DenoisingAE

    # RELEVANT ONLY FOR OCNN 
    PARAMETER_NU = 0.01 # 0.0001 / 0.0005 / 0.001 / 0.005 / 0.01
    PARAMETER_R = 1.0
    HIDDEN_LAYER_SIZE = 12 # 8 / 12 / 16 / 20