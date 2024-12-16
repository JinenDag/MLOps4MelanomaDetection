import os
import numpy as np
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def ModelTesting():

    model_path='models/model.h5'
    csv_filename='predictions.csv'
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    test_set_dir = os.path.join(os.getcwd(), 'real_test_set')
    image_names = [f for f in os.listdir(test_set_dir) if os.path.isfile(os.path.join(test_set_dir, f))]

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Predicted Label'])

        for image_name in image_names:
            image_path = os.path.join(test_set_dir, image_name)

            test_image = load_img(image_path, target_size=(124, 124))
            test_image = img_to_array(test_image) / 255.0
            test_image = np.expand_dims(test_image, axis=0)

            result = model.predict(test_image)
            predicted_label = 1 if result[0][0] > 0.5 else 0

            writer.writerow([image_name, predicted_label])

    print(f"Predictions saved to {csv_filename}")

if __name__ == "__main__":
    ModelTesting()
