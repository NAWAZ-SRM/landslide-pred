import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

# Load the dataset
df = pd.read_csv("./dataset_ordinality_added.csv")  # Replace with the correct path to your dataset

# Encode categorical columns
label_encoders = {}
for col in ['landslide_trigger', 'landslide_region', 'landslide_category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders for later use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Define features and target
X = df[['landslide_trigger', 'landslide_region']]  # Features
y_category = df['landslide_category']  # Target for landslide_category

# Split the dataset for landslide_category
X_train_category, X_test_category, y_train_category, y_test_category = train_test_split(X, y_category, test_size=0.2,
                                                                                        random_state=42)

# Initialize XGBoost Classifier for landslide_category
xgb_classifier_category = XGBClassifier(eval_metric='mlogloss')

# Train the model for landslide_category
xgb_classifier_category.fit(X_train_category, y_train_category)

# Save the model
joblib.dump(xgb_classifier_category, 'xgb_category_model.pkl')

# Predict on the test set
y_pred_category = xgb_classifier_category.predict(X_test_category)

# Calculate performance metrics
accuracy = accuracy_score(y_test_category, y_pred_category)
precision = precision_score(y_test_category, y_pred_category, average='weighted')
recall = recall_score(y_test_category, y_pred_category, average='weighted')
f1 = f1_score(y_test_category, y_pred_category, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_category, y_pred_category)
print("Confusion Matrix:")
print(conf_matrix)

# Create classification report with actual unique classes
unique_classes = y_test_category.unique()
# Create the classification report using the unique classes from the test set
print("Classification Report:")
print(classification_report(y_test_category, y_pred_category, target_names=[f'Class {i}' for i in unique_classes]))


# ===================== Testing the Model =====================

# Load the trained model
xgb_classifier_category = joblib.load('xgb_category_model.pkl')

# Load the label encoder for landslide_category
label_encoders = joblib.load('label_encoders.pkl')  # Make sure to save label_encoders during training


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


# Example of how to test the model
landslide_trigger_input = 'rain'  # Replace with actual input
landslide_region_input = 'Rainfall-Related'  # Replace with actual input

predicted_category = predict_landslide_category(landslide_trigger_input, landslide_region_input)
print(f"Predicted Landslide Category: {predicted_category}")
