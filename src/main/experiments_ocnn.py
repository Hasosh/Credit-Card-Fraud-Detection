import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from dl_models import *
from data_loader import MyDataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from evaluation import ModelEvaluator
import wandb


def quantile_loss(r, y_hat, nu):
    """
    3rd term in Eq (4) of the original paper
    :param r: bias of hyperplane
    :param y_hat: data / output we're operating on
    :param nu: parameter between [0, 1] controls trade-off between maximizing the distance of the hyperplane from
        the origin and the number of data points permitted to cross the hyper-plane (false positives) (default 1e-2)
    :return: the loss function value
    """
    return (1 / nu) * torch.mean(torch.relu(r - y_hat)) - r

def custom_ocnn_loss(model, r, nu):
    """
    Compute the OC-NN loss.
    :param y_hat: The predicted values from the model.
    :return: The computed loss value.
    """
    def loss(y_hat):
        w1_norm = 0.5 * torch.norm(model.dense_out1.weight)**2
        w2_norm = 0.5 * torch.norm(model.out2.weight)**2
        q_loss = quantile_loss(r, y_hat, nu)
        return w1_norm + w2_norm + q_loss
    return loss

batch_size_all = [128, 256]
hidden_layer_size_all = [20, 16, 12, 8]
nu_all = [0.0001, 0.0005, 0.001, 0.005, 0.01]

for batch_size in batch_size_all:
    for hidden_layer_size in hidden_layer_size_all:
        for nu in nu_all:

            print("Using batch size: ", batch_size)
            print("Using hidden layer size: ", hidden_layer_size)
            print("Using nu: ", nu)

            # Initialize a new run
            if Config.USE_WANDB:
                wandb.init(project="dl-anomaly-ocnn", entity="hasan-evci", config={
                    "random_seed": Config.RANDOM_SEED,
                    "validate_size": Config.VALIDATE_SIZE,
                    "batch_size": batch_size,
                    "epochs": Config.EPOCHS,
                    "learning_rate": Config.LEARNING_RATE,
                    "patience": Config.PATIENCE,
                    "min_delta": Config.MIN_DELTA,
                    "hidden_layer_size": hidden_layer_size,
                    "nu": nu,
                    "r": Config.PARAMETER_R
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

            # Set the input dimension
            input_dim=X_train_.shape[1]

            # Model
            model = OneClassNN(input_dim, hidden_layer_size)
            print(model)

            # using GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device = "cpu"
            model = model.to(device)
            print("Device used: ", device)

            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

            ### Training Loop
            r = Config.PARAMETER_R
            best_normal_percentage = 0
            counter = 0
            for epoch in range(Config.EPOCHS):
                model.train()
                train_loss = 0.0
                for data in train_loader:  # Assuming train_loader yields batches of data
                    inputs, _ = data
                    inputs = inputs.to(device)
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = custom_ocnn_loss(model, r, nu)(outputs)
                    loss.backward()
                    optimizer.step()

                    # Update r based on quantile of outputs
                    with torch.no_grad():
                        r = torch.quantile(outputs, nu).item()
                    
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                print(f'Epoch {epoch+1}/{Config.EPOCHS}, Loss: {train_loss}')

                model.eval()
                normal_count = 0
                total_count = 0

                with torch.no_grad():
                    for data in val_loader:
                        inputs, _ = data
                        inputs = inputs.to(device)

                        outputs = model(inputs)
                        normal_indicator = torch.sign(outputs - r)  # Apply the threshold condition
                        normal_count += torch.sum(normal_indicator > 0).item()  # Count normal instances
                        total_count += inputs.size(0)
                
                normal_percentage = (normal_count / total_count) * 100
                print(f'Epoch {epoch+1}/{Config.EPOCHS}, Validation: {normal_count} out of {total_count} instances are normal ({normal_percentage:.2f}%)')

                # Log metrics to W&B
                if Config.USE_WANDB:
                    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_performance": normal_percentage/100})

                # Check for improvement for early stopping
                if normal_percentage - best_normal_percentage > Config.MIN_DELTA:
                    best_normal_percentage = normal_percentage
                    counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), 'best_ocnn.pth')
                else:
                    counter += 1

                if counter >= Config.PATIENCE:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

            ### Evaluation 
                
            # Convert the transformed test set to a PyTorch tensor
            X_test_tensor = torch.Tensor(X_test).to(device)

            # Get the model's anomaly scores on the test set
            model.eval()
            with torch.no_grad():
                test_scores = model(X_test_tensor).cpu().numpy()
            test_scores = np.reshape(test_scores, -1)

            import matplotlib.pyplot as plt
            import numpy as np

            # Histogram of decision scores
            normal_scores = test_scores[y_test == 0]
            anomaly_scores = test_scores[y_test == 1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='green')
            ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
            ax.axvline(r, color='black', linestyle='dashed', linewidth=2, label=f'r={r}')
            ax.set_xlabel('Decision Scores')
            ax.set_ylabel('Frequency')
            ax.set_title('Histogram of Decision Scores')
            ax.legend()
            plt.show()

            # Log the plot to wandb
            if Config.USE_WANDB:
                wandb.log({"Histogram decision scores": wandb.Image(fig)})

            # Set the threshold
            threshold = r

            # Classify as anomaly if the score is below the threshold
            y_pred = (test_scores < threshold).astype(int)
            print("Number of outliers: ", sum(y_pred))

            # compute metrics
            evaluator = ModelEvaluator(y_test, y_pred, -test_scores)
            metrics = evaluator.full_report()
            if Config.USE_WANDB:
                wandb.log(metrics)