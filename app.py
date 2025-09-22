import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import altair as alt

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Mind@Work Dashboard")

# Add a suitable background color to the entire application
st.markdown(
    """
    <style>
    /* Overall app background color */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Style the sidebar navigation radio buttons to look like buttons */
    [data-testid="stSidebar"] .stRadio > div > label {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #d3d3d3;
        transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out;
        font-size: 1.1rem;
        font-weight: bold;
        color: #333333;
    }
    /* Hover effect for the buttons */
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: #e6e6e6;
        border-color: #a9a9a9;
    }
    /* Style for the selected button */
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:hover {
        background-color: #e6e6e6;
        border-color: #a9a9a9;
    }
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] {
        background-color: #e6e6e6;
        border-color: #a9a9a9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set a consistent style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- ⚙️ Load Data ---
@st.cache_data
def load_data():
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv("Cleaned_remote_work.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'Cleaned_remote_work.csv' not found. Please ensure the file is in the correct directory.")
        return None

df = load_data()
if df is None:
    st.stop()

# --- Pre-computation for Dashboard ---
# This part is crucial for the tabs to work efficiently.
# We process the data and train the model once, then pass the results to the tabs.
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
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(X[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
encoded_df.index = X.index

# Combine numerical and encoded categorical data
X_final = pd.concat([X[numerical_features], encoded_df], axis=1)

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# --- Page Functions ---

def show_about_page():
    """Displays the About page with an infographic layout."""
    st.title("Mind@Work: A Data-Driven Approach to Workplace Wellness")
    st.markdown("---")

    # Slide 1: Title Slide
    st.header("Project Details")
    st.markdown("Mind@Work: A Data-Driven Approach to Workplace Wellness is a project to understand employee well-being in remote and hybrid work environments.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🏷️ Project Name")
        st.info("Mind@Work")
    with col2:
        st.markdown("### 📅 Duration")
        st.info("7 weeks")
    with col3:
        st.markdown("### 💰 Budget")
        st.info("154,913 SEK")
    with col4:
        st.markdown("### 🏦 Funding")
        st.info("Public Health Agency of Sweden")

    st.markdown("---")
    
    # Slide 2: Project Justification
    st.header("Project Justification")
    st.markdown("### Why This Project is Important")
    st.markdown("""
    Employee well-being is a critical issue. Mental health challenges among workers lead to substantial costs and reduced productivity. This project addresses this by providing data-driven insights to help organizations support their teams more effectively.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="The Problem", value="Lack of Data on Work Conditions' Impact")
    with col2:
        st.metric(label="Global Cost of Mental Health Disorders", value="$1 Trillion", delta="Lost Workdays Annually")
    with col3:
        st.metric(label="Our Solution", value="Data-Driven Tool")
    
    st.info("This project aligns with the strategic goals of the **OECD**, **Swedish Ministry of Employment**, and **Swedish eHealth Agency**.")
    st.markdown("---")

    # Slide 3: Project Scope & Objectives
    st.header("Project Scope & Objectives")
    st.markdown("### What We Will Deliver")
    st.markdown("""
    The primary objective is to design and implement a web-based, data-driven tool to provide actionable insights.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ In Scope")
        st.markdown("""
        - Using a synthetic Kaggle dataset for analysis.
        - Creating a functional prototype with a dashboard.
        - Generating a final report and presentation.
        """)
    with col2:
        st.subheader("❌ Out of Scope")
        st.markdown("""
        - Using real, sensitive employee data.
        - Building a full commercial product.
        - Providing long-term maintenance or clinical use.
        """)
    st.markdown("---")

    # Slide 4: Project Organization & Risk
    st.header("Team and Contingency Plan")
    st.markdown("""
    The team will follow an agile approach with rotating leadership to ensure collaboration and adaptability.
    """)
    st.subheader("Key Risks & Mitigation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### ⏳ Time Pressure")
        st.warning("Mitigated with a buffer period and weekly check-ins.")
    with col2:
        st.markdown("#### 📉 Poor Data Quality")
        st.warning("Mitigated with thorough data cleaning and backup datasets.")
    with col3:
        st.markdown("#### 👤 Team Unavailability")
        st.warning("Mitigated with knowledge sharing and flexible task allocation.")

    st.markdown("---")


def show_dashboard_page():
    """Displays the main dashboard with tabs."""
    st.title("Remote Work Analytics Dashboard 📊")
    st.markdown("This dashboard provides a comprehensive analysis of remote work data, from descriptive insights to predictive and prescriptive scenarios.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])

    with tab1:
        st.header("1. Descriptive Analytics: What Happened?")
        st.markdown("This section provides a summary of key metrics and data distributions.")

        # Altair chart for Work Location Distribution
        st.subheader("Work Location Distribution")
        chart1 = alt.Chart(df).mark_bar().encode(
            x=alt.X('Work_Location', title='Work Location'),
            y=alt.Y('count()', title='Number of Employees'),
            tooltip=['Work_Location', 'count()']
        ).properties(
            title="Count of Employees by Work Location"
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)
        st.markdown("---")

        # Altair chart for Weekly Hours Worked Distribution
        st.subheader("Weekly Hours Worked Distribution")
        chart2 = alt.Chart(df).mark_area(
            line={'color':'darkgreen'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0),
                       alt.GradientStop(color='darkgreen', offset=1)]
            )
        ).encode(
            x=alt.X('Hours_Worked_Per_Week', bin=alt.Bin(maxbins=20), title='Hours Worked Per Week'),
            y=alt.Y('count()', title='Frequency'),
            tooltip=[alt.Tooltip('Hours_Worked_Per_Week', bin=True), 'count()']
        ).properties(
            title="Distribution of Weekly Hours Worked"
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)
        st.markdown("---")

    with tab2:
        st.header("2. Diagnostic Analytics: Why Did It Happen?")
        st.markdown("This section explores relationships to identify the root causes of outcomes, such as satisfaction.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Job Satisfaction by Work Location")
            # Using a more compact chart without columns
            chart3 = alt.Chart(df).mark_bar().encode(
                x=alt.X('Satisfaction_with_Remote_Work', sort=['Satisfied', 'Neutral', 'Unsatisfied'], title='Satisfaction Level'),
                y=alt.Y('count()', title='Number of Employees'),
                color=alt.Color('Work_Location', title='Work Location'),
                tooltip=['Satisfaction_with_Remote_Work', 'Work_Location', 'count()']
            ).properties(
                title="Job Satisfaction by Work Location"
            ).interactive()
            st.altair_chart(chart3, use_container_width=True)
        
        with col2:
            st.subheader("Work-Life Balance vs. Satisfaction")
            # New chart showing the relationship between work-life balance and satisfaction
            chart4 = alt.Chart(df).mark_bar().encode(
                x=alt.X('Work_Life_Balance_Rating', bin=False, title='Work-Life Balance Rating'),
                y=alt.Y('count()', title='Number of Employees'),
                color=alt.Color('Satisfaction_with_Remote_Work', title='Satisfaction Level'),
                tooltip=['Work_Life_Balance_Rating', 'Satisfaction_with_Remote_Work', 'count()']
            ).properties(
                title="Work-Life Balance vs. Satisfaction"
            ).interactive()
            st.altair_chart(chart4, use_container_width=True)
            
        st.markdown("---")


    with tab3:
        st.header("3. Predictive Analytics: What Will Happen?")
        st.markdown("This section uses a machine learning model to predict the satisfaction level based on key factors.")
        st.write(f"**Model Accuracy:** {accuracy:.2f}")
        st.write("This simple model predicts job satisfaction based on hours worked, work-life balance, physical activity, and sleep quality.")
        st.markdown("---")

    with tab4:
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

        # Preprocess the new input data
        input_numerical = input_df[numerical_features]
        input_encoded = encoder.transform(input_df[categorical_features])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_features))
        
        # Combine numerical and encoded categorical data for prediction
        final_input = pd.concat([input_numerical.reset_index(drop=True), input_encoded_df], axis=1)

        # Make a prediction
        try:
            prediction = model.predict(final_input)
            prediction_proba = model.predict_proba(final_input)
            
            if prediction[0] == 1:
                st.success(f"Based on the inputs, the model predicts the individual will likely be **Satisfied** with remote work.")
                st.write(f"Probability of being satisfied: {prediction_proba[0][1]*100:.2f}%")
            else:
                st.warning(f"Based on the inputs, the model predicts the individual will likely be **Unsatisfied/Neutral** with remote work.")
                st.write(f"Probability of being unsatisfied/neutral: {prediction_proba[0][0]*100:.2f}%")
        except Exception as e:
            st.error(f"Prediction error: {e}")
        st.markdown("---")


# --- Main Application Logic ---
# Use a radio button in the sidebar for navigation
page = st.sidebar.radio("Navigate", ["About", "Dashboard"], help="Select a page to view.")

if page == "About":
    show_about_page()
else:
    show_dashboard_page()
