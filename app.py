import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai

genai.configure(
    api_key="" #API KEY
)

model_gemini = genai.GenerativeModel('gemini-2.0-flash')

model = tf.keras.models.load_model('lung_cancer_model.h5')

class_labels = ['Benign', 'Malignant', 'Normal']

medical_keywords = {'cancer', 'medical', 'lung', 'treatment', 'symptoms', 
                   'diagnosis', 'benign', 'malignant', 'tumor', 'health',
                   'medicine', 'doctor', 'patient', 'hospital', 'disease', 'precaution'}

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def is_medical_question(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in medical_keywords)

st.title("Lung Cancer Diagnosis & Medical Chat Assistant")

st.header("Lung Cancer Prediction")
uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.subheader(f"Prediction: {predicted_class}")

st.header("Medical Chat Assistant")
user_input = st.text_input("Ask a medical question:")

if user_input:
    if is_medical_question(user_input):
        try:
            response = model_gemini.generate_content(
                user_input,
                safety_settings={
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                },
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )
            st.write("Assistant:", response.text)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("I'm specialized in medical and cancer-related information only.")
