import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.h5')

# Define the labels for the classes
class_names = ['Benign', 'Malignant']


# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 224x224 pixels
    image = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    st.write("This image most likely belongs to ",
             class_names[np.argmax(score)],"with a percent confidence of",100 * np.max(score))


# Define the Streamlit app
def main():
    # Set the title of the app
    st.title("Image Classification App")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocess_image(image)



# Run the Streamlit app
if __name__ == '__main__':
    main()

