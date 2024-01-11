# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from dl_models import Autoencoder #, VariationalAutoencoder 
from data_loader import MyDataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from evaluation import ModelEvaluator
import wandb

# Initialize a new run
wandb.init(project="dl-lab", entity="hasan-evci", config={
    "random_seed": Config.RANDOM_SEED,
    "validate_size": Config.VALIDATE_SIZE,
    "batch_size": Config.BATCH_SIZE,
    "epochs": Config.EPOCHS,
    "learning_rate": Config.LEARNING_RATE,
    "model_type": Config.MODEL_TYPE,
    "num_layers": Config.NUM_LAYERS,
    "bottleneck_size": Config.BOTTLENECK_SIZE,
    "activation_func": Config.ACTIVATION_FUNC,
    "patience": Config.PATIENCE,
    "min_delta": Config.MIN_DELTA,
    "loss_function": Config.LOSS_FUNCTION,
    "plot_clipping_value": Config.PLOT_CLIPPING_VALUE,
})


# Set random seed
np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)

# Load data
loader = MyDataLoader()
setup1 = loader.load_setup(Config.DATA_PATH)

X_train = loader.transform_data(setup1['X_train'], is_train=True)
X_test = loader.transform_data(setup1['X_test'], is_train=False)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

y_test = setup1['y_test']

print("Testing Labels Data Shape:", y_test.shape)

# Batching
X_train_, X_validate_ = train_test_split(X_train, 
                                       test_size=Config.VALIDATE_SIZE, 
                                       random_state=Config.RANDOM_SEED)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train_)

# Create a dataset and data loader
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

# Convert validation data to PyTorch tensors
X_val_tensor = torch.Tensor(X_validate_)

# Create a dataset and data loader for validation
val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=Config.BATCH_SIZE)

# Create model
if Config.MODEL_TYPE == 'Autoencoder' or Config.MODEL_TYPE == 'DenoisingAE':
    model = Autoencoder(input_dim=X_train_.shape[1],
                        num_layers=Config.NUM_LAYERS,
                        bottleneck_size=Config.BOTTLENECK_SIZE,
                        activation_func=getattr(nn, Config.ACTIVATION_FUNC))
elif Config.MODEL_TYPE == 'VariationalAutoencoder':
    pass # model = VariationalAutoencoder()
print(model)

# using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device = "cpu"
model = model.to(device)
print("Device used: ", device)

# Define loss function
if Config.LOSS_FUNCTION == 'BCE':
    criterion = nn.BCELoss()
elif Config.LOSS_FUNCTION == 'MSE':
    criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

def add_noise(inputs, noise_factor=0.5): # only needed for denoising autoencoder
    noise = torch.randn_like(inputs) * noise_factor
    return inputs + noise

# Training loop
best_loss = float('inf')
counter = 0
for epoch in range(Config.EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for data in train_loader:
        inputs, targets = data
        optimizer.zero_grad()

        if Config.MODEL_TYPE == 'DenoisingAE':
            noisy_inputs = add_noise(inputs, noise_factor=Config.NOISE_FACTOR)
            noisy_inputs, targets = noisy_inputs.to(device), targets.to(device) 
            outputs = model(noisy_inputs)
        else:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the same device as the model
            outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the same device as the model
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')

    # Log metrics to W&B
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # Check for improvement for early stopping
    if best_loss - val_loss > Config.MIN_DELTA:
        best_loss = val_loss
        counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_autoencoder.pth')
    else:
        counter += 1

    if counter >= Config.PATIENCE:
        print(f'Early stopping triggered after {epoch + 1} epochs')
        break

# Evaluation
    
# Convert the transformed test set to a PyTorch tensor
X_test_tensor = torch.Tensor(X_test).to(device)

# Get the model's reconstruction of the test set
model.eval()
with torch.no_grad():
    reconstructions = model(X_test_tensor).cpu().numpy()

# Calculate the MSE reconstruction loss per row
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

clean = mse[y_test == 0]
fraud = mse[y_test == 1]

# Plotting the distribution of reconstruction loss
fig, ax = plt.subplots(figsize=(6,6))
ax.hist(clean[clean <= Config.PLOT_CLIPPING_VALUE], bins=50, density=True, label="clean", alpha=.6, color="green")
ax.hist(fraud[fraud <= Config.PLOT_CLIPPING_VALUE], bins=50, density=True, label="fraud", alpha=.6, color="red")
plt.title("(Normalized) Distribution of the Reconstruction Loss")
plt.xlabel("Reconstruction error")
plt.legend()
#plt.show()

# Log the plot to wandb
wandb.log({"Reconstruction Loss Distribution": wandb.Image(fig)})

def get_best_threshold(y_true, reconstruction_losses):
    best_f1 = 0
    best_threshold = 0

    # Iterate over a range of possible thresholds
    for threshold in np.linspace(min(reconstruction_losses), max(reconstruction_losses), num=1000):
        
        # Predict anomalies based on the threshold
        y_pred = (mse > threshold).astype(int)
        
        # Calculate F1 score
        current_f1 = f1_score(y_true, y_pred)
        
        # Update the best threshold if current F1 score is higher
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1}")

    return best_threshold

threshold = get_best_threshold(y_test, mse)

outliers = mse > threshold
print("Number of outliers: ", sum(outliers))

y_pred = outliers

# Convert predictions to match y_test labels (1 for anomalies, 0 for normal)
y_pred = y_pred.astype(int)

# compute metrics
evaluator = ModelEvaluator(y_test, y_pred, mse)
metrics = evaluator.basic_report()
#metrics = evaluator.full_report()
wandb.log(metrics)

# Save and upload the model
# wandb.save("model.pth")

# Finish the run
wandb.finish()