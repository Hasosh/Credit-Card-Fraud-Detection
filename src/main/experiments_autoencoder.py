import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from dl_models import Autoencoder
from data_loader import MyDataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from evaluation import ModelEvaluator
import wandb

batch_size_all = [128, 256]
num_layers_all = [4, 6, 8]
bottleneck_size_all = [2, 3, 4, 6, 8]
activation_func_all = ['ELU', 'ReLU', 'Tanh']

for batch_size in batch_size_all:
    for num_layers in num_layers_all:
        for bottleneck_size in bottleneck_size_all:
            for activation_func in activation_func_all:

                # cases when bottleneck size is too high for the number of layers: continue with next configuration
                if num_layers == 6:
                    if bottleneck_size > 4:
                        continue
                if num_layers == 8:
                    if bottleneck_size > 2:
                        continue

                if batch_size==128 and bottleneck_size==2:
                    continue
                if batch_size==128 and bottleneck_size==3 and activation_func=='ELU':
                    continue

                print("Using batch size: ", batch_size)
                print("Using num layers: ", num_layers)
                print("Using bottleneck_size: ", bottleneck_size)
                print("Using activation_func: ", activation_func)

                # Initialize a new run
                wandb.init(project="dl-lab", entity="hasan-evci", name=str(batch_size)+"_"+str(num_layers)+"_"+str(bottleneck_size)+"_"+str(activation_func), config={
                    "random_seed": Config.RANDOM_SEED,
                    "validate_size": Config.VALIDATE_SIZE,
                    "batch_size": batch_size,
                    "epochs": Config.EPOCHS,
                    "learning_rate": Config.LEARNING_RATE,
                    "model_type": Config.MODEL_TYPE,
                    "num_layers": num_layers,
                    "bottleneck_size": bottleneck_size,
                    "activation_func": activation_func,
                    "patience": Config.PATIENCE,
                    "min_delta": Config.MIN_DELTA,
                    "loss_function": Config.LOSS_FUNCTION,
                    "plot_clipping_value": Config.PLOT_CLIPPING_VALUE,
                })


                # Set random seed
                np.random.seed(Config.RANDOM_SEED)
                torch.manual_seed(Config.RANDOM_SEED)

                #### Load data
                loader = MyDataLoader()
                setup = loader.load_setup(Config.DATA_PATH)

                loader.fit_data(setup['X_train'])
                X_train = loader.transform_data(setup['X_train'])
                X_test = loader.transform_data(setup['X_test'])

                print("Training Data Shape:", X_train.shape)
                print("Testing Data Shape:", X_test.shape)

                y_test = setup['y_test']

                print("Testing Labels Data Shape:", y_test.shape)

                # Batching
                X_train_, X_validate_ = train_test_split(X_train, 
                                                    test_size=Config.VALIDATE_SIZE, 
                                                    random_state=Config.RANDOM_SEED)

                # Convert data to PyTorch tensors
                X_train_tensor = torch.Tensor(X_train_)

                # Create a dataset and data loader
                train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

                # Convert validation data to PyTorch tensors
                X_val_tensor = torch.Tensor(X_validate_)

                # Create a dataset and data loader for validation
                val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
                val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

                # Create model
                model = Autoencoder(input_dim=X_train_.shape[1],
                                    num_layers=num_layers,
                                    bottleneck_size=bottleneck_size,
                                    activation_func=getattr(nn, activation_func))

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

                #### Training loop
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
                            inputs = add_noise(inputs, noise_factor=Config.NOISE_FACTOR)

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

                #### Find thresholds
                    
                X_train_tensor = X_train_tensor.to(device)

                # Get the model's reconstruction of the test set
                model.eval()
                with torch.no_grad():
                    reconstructions = model(X_train_tensor).cpu().numpy()

                # Calculate the MSE reconstruction loss per row
                train_mse = np.mean(np.power(X_train_tensor.cpu().numpy() - reconstructions, 2), axis=1)

                threshold1 = np.quantile(train_mse, 0.90)
                threshold2 = np.quantile(train_mse, 0.95)
                threshold3 = np.quantile(train_mse, 0.99)
                threshold4 = np.mean(train_mse) + np.std(train_mse)

                #### Evaluation
                    
                # Convert the transformed test set to a PyTorch tensor
                X_test_tensor = torch.Tensor(X_test).to(device)

                # Get the model's reconstruction of the test set
                model.eval()
                with torch.no_grad():
                    reconstructions = model(X_test_tensor).cpu().numpy()

                # Calculate the MSE reconstruction loss per row
                mse = np.mean(np.power(X_test_tensor.cpu().numpy() - reconstructions, 2), axis=1)

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

                threshold5 = get_best_threshold(y_test, mse) # for comparison

                def full_evaluation(mse, threshold):
                    outliers = mse > threshold
                    print("Number of outliers: ", sum(outliers))

                    # Convert predictions to match y_test labels (1 for anomalies, 0 for normal)
                    y_pred = outliers.astype(int)

                    # compute metrics
                    evaluator = ModelEvaluator(y_test, y_pred, mse)
                    metrics = evaluator.basic_report()
                    #metrics = evaluator.full_report()
                    wandb.log(metrics)

                full_evaluation(mse, threshold1)
                full_evaluation(mse, threshold2)
                full_evaluation(mse, threshold3)
                full_evaluation(mse, threshold4)
                full_evaluation(mse, threshold5)

                # Save and upload the model
                # wandb.save("model.pth")

                # Finish the run
                wandb.finish()