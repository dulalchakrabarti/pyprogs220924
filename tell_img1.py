import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

model = EfficientNetB0(weights='imagenet')
# (You would modify the top layers for your specific crop classes)

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=3)[0]
    
    return decoded_preds

# Usage
filenames=["/media/dc/DC-28122024/wheatcam3/Wheat/image_0463.jpg"]
for filename in filenames:
 predictions = classify_image(filename)
 for i, (imagenet_id, label, prob) in enumerate(predictions):
    print(f"{i+1}: {label} ({prob*100:.2f}%)")
