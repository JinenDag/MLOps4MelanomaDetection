from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import yaml

def DataPreparation():
    # Load configuration parameters from YAML file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    shear_range = config["shear_range"]
    zoom_range = config["zoom_range"]
    horizontal_flip = config["horizontal_flip"]
    batch_size_Data_Generator = config["batch_size_Data_Generator"]

    # Create ImageDataGenerator objects for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')

    # Directory where augmented data will be saved
    augm_dir = os.path.join(base_dir, 'augm_data')

    # Clean up existing augmented data directory
    if os.path.exists(augm_dir):
        shutil.rmtree(augm_dir)

    # Create directories for training and test sets in augmented data folder
    os.makedirs(os.path.join(augm_dir, "training_set"), exist_ok=True)
    os.makedirs(os.path.join(augm_dir, "test_set"), exist_ok=True)

    # Get class names from the original training and test set directories
    training_set_src_dir = os.path.join(data_dir, 'training_set')
    test_set_src_dir = os.path.join(data_dir, 'test_set')

    class_names = sorted(os.listdir(training_set_src_dir))

    # Ensure subdirectories exist in augmented training and test set directories
    for class_name in class_names:
        os.makedirs(os.path.join(augm_dir, "training_set", class_name), exist_ok=True)
        os.makedirs(os.path.join(augm_dir, "test_set", class_name), exist_ok=True)

    # Prepare test set generator
    test_set_dest_dir = os.path.join(augm_dir, "test_set")
    test_generator = test_datagen.flow_from_directory(
        directory=test_set_src_dir,
        target_size=(124, 124),
        batch_size=batch_size_Data_Generator,
        class_mode='binary',
        shuffle=False,
        save_to_dir=test_set_dest_dir,
        save_prefix='aug_test_',
        save_format='jpeg'
    )

    # Prepare training set generator
    training_set_dest_dir = os.path.join(augm_dir, 'training_set')
    train_generator = train_datagen.flow_from_directory(
        directory=training_set_src_dir,
        target_size=(124, 124),
        batch_size=batch_size_Data_Generator,
        class_mode='binary',
        shuffle=False,
        save_to_dir=training_set_dest_dir,
        save_prefix='aug_train_',
        save_format='jpeg'
    )

    # Generate and save a few batches to create augmented images
    for _ in range(2):  # Adjust the number of batches to generate
        next(train_generator)
        next(test_generator)

    print("Data preparation complete.")
    return train_generator, test_generator

if __name__ == '__main__':
    print("Data preparation ...")
    train_generator, test_generator = DataPreparation()
