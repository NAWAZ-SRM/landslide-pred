import pandas as pd
import joblib
import streamlit as st
from xgboost import XGBClassifier

# Load the trained model and label encoders
xgb_classifier_category = joblib.load('xgb_category_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')


def predict_landslide_category(landslide_trigger, landslide_region):
    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'landslide_trigger': [landslide_trigger],
        'landslide_region': [landslide_region]
    })

    # Encode the input data
    for col in ['landslide_trigger', 'landslide_region']:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])

    # Make prediction
    prediction = xgb_classifier_category.predict(input_data)

    # Decode the prediction to original label
    predicted_category = label_encoders['landslide_category'].inverse_transform(prediction)

    return predicted_category[0]  # Return the first predicted value


# Streamlit App
st.title("Landslide Category Prediction")
st.write("This app predicts the category of landslide based on trigger and region.")

# Input fields
landslide_trigger_input = st.selectbox(
    "Select Landslide Trigger:",
    label_encoders['landslide_trigger'].classes_
)

landslide_region_input = st.selectbox(
    "Select Landslide Region:",
    label_encoders['landslide_region'].classes_
)

# Button to make prediction
if st.button("Predict Landslide Category"):
    predicted_category = predict_landslide_category(landslide_trigger_input, landslide_region_input)
    st.success(f"Predicted Landslide Category: **{predicted_category}**")
