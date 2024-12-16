import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def ModelValidation():
    # Define the folder for saving plots
    model_folder = "models"
    os.makedirs(model_folder, exist_ok=True)

    # Load the trained model
    model_path = os.path.join(model_folder, "model.h5")
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Load the training history
    history_path = os.path.join(model_folder, "training_history.json")
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"Loaded training history from {history_path}")

    # Image data generator setup
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Ensure directories exist for the datasets
    train_set_dir = "data/training_set"
    test_set_dir = "data/test_set"

    if not os.path.exists(train_set_dir):
        print(f"Training set directory does not exist: {train_set_dir}")
        return
    if not os.path.exists(test_set_dir):
        print(f"Test set directory does not exist: {test_set_dir}")
        return

    # Load the datasets
    training_set = train_datagen.flow_from_directory(
        directory=train_set_dir,
        target_size=(124, 124),
        batch_size=32,
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        directory=test_set_dir,
        target_size=(124, 124),
        batch_size=32,
        class_mode='categorical'
    )

    # Evaluate the model
    result_test = model.evaluate(test_set, steps=test_set.samples // test_set.batch_size, verbose=1)
    print(f"Test-set classification accuracy: {result_test[1]:.2%}")
    print(f"Test-set loss: {result_test[0]:.4f}")

    result_train = model.evaluate(training_set, steps=training_set.samples // training_set.batch_size, verbose=1)
    print(f"Train-set classification accuracy: {result_train[1]:.2%}")
    print(f"Train-set loss: {result_train[0]:.4f}")

    # Plot accuracy
    plt.figure()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    accuracy_plot_path = os.path.join(model_folder, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy plot saved to {accuracy_plot_path}")
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    loss_plot_path = os.path.join(model_folder, "loss_plot.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    plt.close()

if __name__ == "__main__":
    print("Model Validation ...")
    ModelValidation()
