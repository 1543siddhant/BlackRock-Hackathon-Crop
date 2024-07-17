import streamlit as st
import numpy as np
import joblib  # Change from pickle to joblib

# Load the models
model = joblib.load('model.pkl')  # Change from pickle.load to joblib.load
sc = joblib.load('standscaler.pkl')  # Change from pickle.load to joblib.load
ms = joblib.load('minmaxscaler.pkl')  # Change from pickle.load to joblib.load

# Mapping for crop types
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# App layout
st.title("Crop Recommendation System")

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Crop Prediction"])

# Home Page
if app_mode == "Home":
    st.header("Welcome to the Agricultural Intelligence System")
    image_path = "home.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    ## Features:
    1. Crop Recommendation based on soil and weather conditions.
    """)

# Crop Prediction Page
elif app_mode == "Crop Prediction":
    st.header("Crop Recommendation")

    N = st.slider('Nitrogen', min_value=0, max_value=100, step=1)
    P = st.slider('Phosphorus', min_value=0, max_value=100, step=1)
    K = st.slider('Potassium', min_value=0, max_value=100, step=1)
    temp = st.slider('Temperature (°C)', min_value=-10.0, max_value=50.0, step=0.1)
    humidity = st.slider('Humidity (%)', min_value=0.0, max_value=100.0, step=0.1)
    ph = st.slider('pH', min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.slider('Rainfall (mm)', min_value=0.0, max_value=500.0, step=0.1)

    if st.button("Predict Crop"):
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        st.success(result)

# # About Page
# elif app_mode == "About":
#     st.header("About")
#     st.markdown("""
#     ## About the Project
#     This project aims to assist farmers in making informed decisions about crop selection through advanced machine learning models.
#     """)
