import OneClassSVM
# import kNN
# import IsolationForest
# import Autoencoder


if __name__ == "__main__":
    print("Dear User, please choose one of the following models you want to use:")
    print("1. One-Class SVM")
    print("2. kNN")
    print("3. Isolation Forest")
    print("4. Autoencoder")
    print("Now please enter the number of the model you want to use: ")

    # Get a number from the input of the user
    number = int(input())

    if number == 1:
        print("You have chosen the One-Class SVM model.")
        OneClassSVM.main()
    elif number == 2:
        print("You have chosen the kNN model.")
    elif number == 3:
        print("You have chosen the Isolation Forest model.")
    elif number == 4:
        print("You have chosen the Autoencoder model.")
    else:
        print("Please enter a valid number.")

    print("Thank you for using our program.")