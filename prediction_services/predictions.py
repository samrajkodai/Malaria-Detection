import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import yaml

def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    result=np.argmax(preds, axis=1)

    if result==0:
        result="Parasite"
    else:
        result="Uninfected"
        
    print(result)     # Convert to string
    return result