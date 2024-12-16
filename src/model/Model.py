from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout

def Model_init(num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(124, 124, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())  # Converts 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    return model

if __name__ == "__main__":
    num_classes = 2
    print("Model init...")
    model = Model_init(num_classes)

    model_path = "models/initialized_model.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")
