import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Set a consistent style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- ⚙️ Load Data ---
@st.cache_data
def load_data():
    """Loads and preprocesses the dataset."""
    try:
        # The filename is accessible as a special token
        df = pd.read_csv("Cleaned_remote_work.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'Cleaned_remote_work.csv' not found. Please ensure the file is in the correct directory.")
        return None

df = load_data()
if df is None:
    st.stop()

st.title("Remote Work Analytics Dashboard 📊")
st.markdown("---")

# --- 📈 Descriptive Analytics ---
st.header("1. Descriptive Analytics: What Happened?")
st.markdown("This section provides a summary of key metrics and data distributions.")

st.subheader("Work Location Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(x='Work_Location', data=df, ax=ax1, order=df['Work_Location'].value_counts().index)
ax1.set_title("Count of Employees by Work Location")
ax1.set_xlabel("Work Location")
ax1.set_ylabel("Number of Employees")
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)
st.markdown("---")

st.subheader("Weekly Hours Worked Distribution")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.histplot(df['Hours_Worked_Per_Week'], bins=20, kde=True, ax=ax2)
ax2.set_title("Distribution of Weekly Hours Worked")
ax2.set_xlabel("Hours Worked Per Week")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)
st.markdown("---")

# --- 🔍 Diagnostic Analytics ---
st.header("2. Diagnostic Analytics: Why Did It Happen?")
st.markdown("This section explores relationships to identify the root causes of outcomes, such as satisfaction.")

st.subheader("Job Satisfaction by Work Location")
fig3, ax3 = plt.subplots(figsize=(10, 7))
sns.countplot(x='Satisfaction_with_Remote_Work', hue='Work_Location', data=df, ax=ax3,
              order=['Satisfied', 'Neutral', 'Unsatisfied'])
ax3.set_title("Job Satisfaction by Work Location")
ax3.set_xlabel("Satisfaction Level")
ax3.set_ylabel("Number of Employees")
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)
st.markdown("---")

# --- 🎯 Predictive Analytics ---
st.header("3. Predictive Analytics: What Will Happen?")
st.markdown("This section uses a machine learning model to predict the satisfaction level based on key factors.")

# Data preprocessing for the model
df_model = df.copy()
df_model = df_model.dropna(subset=['Hours_Worked_Per_Week', 'Work_Life_Balance_Rating', 'Physical_Activity', 'Sleep_Quality', 'Satisfaction_with_Remote_Work'])

# Simplify 'Satisfaction_with_Remote_Work' to a binary target variable
df_model['Satisfaction_Binary'] = df_model['Satisfaction_with_Remote_Work'].apply(
    lambda x: 1 if x == 'Satisfied' else 0
)

# Selecting features and target variable
numerical_features = ['Hours_Worked_Per_Week', 'Work_Life_Balance_Rating']
categorical_features = ['Physical_Activity', 'Sleep_Quality']
X = df_model[numerical_features + categorical_features]
y = df_model['Satisfaction_Binary']

# Encode categorical variables
# We fit the encoder on the entire dataset to ensure all possible categories are known
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(X[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
encoded_df.index = X.index # Align indices for concatenation

# Combine numerical and encoded categorical data
X_final = pd.concat([X[numerical_features], encoded_df], axis=1)

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000) # Added max_iter to prevent convergence warnings
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2f}")
st.write("This simple model predicts job satisfaction based on hours worked, work-life balance, physical activity, and sleep quality.")
st.markdown("---")

# --- 💡 Prescriptive Analytics ---
st.header("4. Prescriptive Analytics: What Should We Do?")
st.markdown("This section provides recommendations by allowing you to simulate different scenarios and see their potential impact on satisfaction.")

st.subheader("What-If Scenario Simulator")
st.write("Adjust the sliders below to see how these factors might influence an individual's satisfaction.")

# Use unique values from the dataset for selectbox options
physical_activity_options = sorted(df['Physical_Activity'].dropna().unique())
sleep_quality_options = sorted(df['Sleep_Quality'].dropna().unique())

# Create input widgets for the user
hours_worked = st.slider("Hours Worked Per Week", min_value=20, max_value=80, value=40)
work_life_balance = st.slider("Work-Life Balance Rating (1-5)", min_value=1, max_value=5, value=3)
physical_activity = st.selectbox("Physical Activity", physical_activity_options)
sleep_quality = st.selectbox("Sleep Quality", sleep_quality_options)

# Create a DataFrame for the prediction
input_df = pd.DataFrame({
    'Hours_Worked_Per_Week': [hours_worked],
    'Work_Life_Balance_Rating': [work_life_balance],
    'Physical_Activity': [physical_activity],
    'Sleep_Quality': [sleep_quality]
})

# Preprocess the new input data using the same numerical and categorical features
input_numerical = input_df[numerical_features]
input_encoded = encoder.transform(input_df[categorical_features])
input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Combine numerical and encoded categorical data for prediction
final_input = pd.concat([input_numerical.reset_index(drop=True), input_encoded_df], axis=1)

# Make a prediction
try:
    prediction = model.predict(final_input)
    prediction_proba = model.predict_proba(final_input)
    
    # Display the result
    if prediction[0] == 1:
        st.success(f"Based on the inputs, the model predicts the individual will likely be **Satisfied** with remote work.")
        st.write(f"Probability of being satisfied: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.warning(f"Based on the inputs, the model predicts the individual will likely be **Unsatisfied/Neutral** with remote work.")
        st.write(f"Probability of being unsatisfied/neutral: {prediction_proba[0][0]*100:.2f}%")
except Exception as e:
    st.error(f"Prediction error: {e}")
