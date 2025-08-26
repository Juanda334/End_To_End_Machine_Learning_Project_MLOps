import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, file_name):
        self.file_name = file_name
        
    def predict(self):
        # Load the pre-trained model
        model = load_model(os.path.join("artifacts", "training", "trained_model.h5"))
        
        imagename = self.file_name
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        results = np.argmax(model.predict(test_image), axis = 1)
        print(results)
        
        if results[0] == 1:
            prediction = 'Healthy'
            return [{"image": prediction}]
        else:
            prediction = 'Coccidiosis'
            return [{"image": prediction}]