import json
import yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def Train():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load the pre-initialized model
    model_path = "models/initialized_model.h5"
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        directory="data/training_set",
        target_size=(124, 124),
        batch_size=config["batch_size"],
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        directory="data/test_set",
        target_size=(124, 124),
        batch_size=config["batch_size"],
        class_mode='categorical'
    )

    batch_size = config["batch_size"]
    optimizer = config["optimizer"]
    epochs = config["epochs"]
    lr = config["lr"]

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        training_set,
        epochs=epochs,
        validation_data=test_set
    )

    # Save the model
    model.save("models/model.h5")
    print("Model saved to models/model.h5")

    # Save the history as a JSON file
    history_path = "models/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    print("Training the model ...")
    Train()
