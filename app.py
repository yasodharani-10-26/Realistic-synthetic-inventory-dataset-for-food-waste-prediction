import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI & Social Media Impact",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 20px;
    color: #666;
    margin-top: 0px;
}

.section-title {
    font-size: 28px;
    font-weight: 650;
}

div[data-testid="stMetric"] {
    background-color: #f7f9fc;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">🧠 AI & Social Media Impact</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Student Health & Mental Health Prediction Dashboard</p>',
    unsafe_allow_html=True
)

st.write(
    "This Machine Learning project explores the relationship between "
    "social media usage, AI tool usage, sleep, physical activity, "
    "physical health, and student mental health."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write("""
    **Project:**  
    AI & Social Media Impact

    **ML Task:**  
    Mental Health Score Prediction

    **Algorithm:**  
    Random Forest Regression

    **Dataset:**  
    Student Health & Social Media Dataset
    """)

    st.divider()

    st.header("🎯 Target Variable")

    st.info(
        "The model predicts the student's "
        "**Mental Health Score**."
    )

    st.divider()

    st.caption(
        "⚠️ This application is an educational Machine Learning "
        "project and should not be used as a medical diagnosis."
    )


# ============================================================
# LOAD DATASET
# ============================================================

FILE_PATH = "AI_SocialMedia_Student_Dataset.csv"

try:

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "srisyra02/ai-and-social-media-impact-student-health-and-grades",
        FILE_PATH
    )

except Exception as e:

    st.error(f"❌ Unable to load dataset: {e}")
    st.stop()


# ============================================================
# REQUIRED COLUMNS CHECK
# ============================================================

required_columns = [
    "Student_ID",
    "Age",
    "Gender",
    "Education_Level",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Mental_Health_Score",
    "Physical_Health_Score"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns in dataset: {missing_columns}"
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df = df.copy()

numeric_columns = [
    "Age",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Mental_Health_Score",
    "Physical_Health_Score"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=required_columns
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown(
    '<p class="section-title">📊 Dataset Overview</p>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👨‍🎓 Students",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "📋 Features",
        "10"
    )

with col3:
    st.metric(
        "🧠 Avg Mental Health",
        f"{df['Mental_Health_Score'].mean():.2f}"
    )

with col4:
    st.metric(
        "📱 Avg Social Media",
        f"{df['Daily_Social_Media_Hours'].mean():.2f} hrs"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "📊 Data Analysis",
    "🤖 ML Model",
    "🎯 Prediction"
])


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:

    st.subheader("📌 Project Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### What does this project study?")

        st.write("""
        This project investigates how digital habits and lifestyle
        factors are associated with student mental health.

        The analysis focuses on:

        - 📱 Daily social media usage
        - 🤖 Daily AI tool usage
        - 😴 Sleep duration
        - 🏃 Physical activity
        - ❤️ Physical health
        - 👤 Age, gender and education level
        """)

    with col2:

        st.write("### Machine Learning Workflow")

        st.write("""
        **1. Data Collection**  
        Kaggle student dataset

        **2. Data Preprocessing**  
        Cleaning and categorical encoding

        **3. Feature Selection**  
        Lifestyle and demographic factors

        **4. Model Training**  
        Random Forest Regression

        **5. Evaluation**  
        MAE and R² Score

        **6. Prediction**  
        Student Mental Health Score
        """)

    st.divider()

    st.subheader("📋 Sample Student Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# TAB 2 — DATA ANALYSIS
# ============================================================

with tab2:

    st.subheader("📊 Exploratory Data Analysis")

    st.write(
        "The following charts show relationships between lifestyle "
        "factors and mental health scores."
    )

    # --------------------------------------------------------
    # SOCIAL MEDIA
    # --------------------------------------------------------

    st.write("### 📱 Social Media Usage vs Mental Health")

    social_data = df.copy()

    social_data["Social_Media_Group"] = pd.cut(
        social_data["Daily_Social_Media_Hours"],
        bins=[-np.inf, 2, 4, 6, 8, np.inf],
        labels=[
            "0–2 hours",
            "2–4 hours",
            "4–6 hours",
            "6–8 hours",
            "8+ hours"
        ]
    )

    social_avg = (
        social_data
        .groupby(
            "Social_Media_Group",
            observed=False
        )["Mental_Health_Score"]
        .mean()
    )

    st.bar_chart(social_avg)

    # --------------------------------------------------------
    # AI USAGE
    # --------------------------------------------------------

    st.write("### 🤖 AI Tool Usage vs Mental Health")

    ai_data = df.copy()

    ai_data["AI_Usage_Group"] = pd.cut(
        ai_data["Daily_AI_Tool_Usage_Hours"],
        bins=[-np.inf, 1, 2, 4, 8, np.inf],
        labels=[
            "0–1 hour",
            "1–2 hours",
            "2–4 hours",
            "4–8 hours",
            "8+ hours"
        ]
    )

    ai_avg = (
        ai_data
        .groupby(
            "AI_Usage_Group",
            observed=False
        )["Mental_Health_Score"]
        .mean()
    )

    st.bar_chart(ai_avg)

    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    st.write("### 😴 Sleep Hours vs Mental Health")

    sleep_data = df.copy()

    sleep_data["Sleep_Group"] = pd.cut(
        sleep_data["Sleep_Hours"],
        bins=[-np.inf, 4, 6, 8, 10, np.inf],
        labels=[
            "<4 hours",
            "4–6 hours",
            "6–8 hours",
            "8–10 hours",
            "10+ hours"
        ]
    )

    sleep_avg = (
        sleep_data
        .groupby(
            "Sleep_Group",
            observed=False
        )["Mental_Health_Score"]
        .mean()
    )

    st.bar_chart(sleep_avg)

    # --------------------------------------------------------
    # PHYSICAL ACTIVITY
    # --------------------------------------------------------

    st.write("### 🏃 Physical Activity vs Mental Health")

    activity_data = df.copy()

    activity_data["Activity_Group"] = pd.cut(
        activity_data["Physical_Activity_Hours"],
        bins=[-np.inf, 1, 2, 4, 8, np.inf],
        labels=[
            "0–1 hour",
            "1–2 hours",
            "2–4 hours",
            "4–8 hours",
            "8+ hours"
        ]
    )

    activity_avg = (
        activity_data
        .groupby(
            "Activity_Group",
            observed=False
        )["Mental_Health_Score"]
        .mean()
    )

    st.bar_chart(activity_avg)

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔗 Correlation Analysis")

    correlation_columns = [
        "Age",
        "Daily_Social_Media_Hours",
        "Daily_AI_Tool_Usage_Hours",
        "Sleep_Hours",
        "Physical_Activity_Hours",
        "Mental_Health_Score",
        "Physical_Health_Score"
    ]

    correlation = (
        df[correlation_columns]
        .corr()
        .round(2)
    )

    st.dataframe(
        correlation,
        use_container_width=True
    )

    st.caption(
        "Correlation values closer to +1 or -1 indicate stronger "
        "linear relationships. Correlation does not prove causation."
    )


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

features = [
    "Age",
    "Gender",
    "Education_Level",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Physical_Health_Score"
]

target = "Mental_Health_Score"


model_df = df[
    features + [target]
].copy()


# Encode categorical variables

model_df = pd.get_dummies(
    model_df,
    columns=[
        "Gender",
        "Education_Level"
    ],
    drop_first=True
)


# Remove missing values

model_df = model_df.dropna()


X = model_df.drop(
    columns=[target]
)

y = model_df[target]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# MODEL METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# TAB 3 — ML MODEL
# ============================================================

with tab3:

    st.subheader("🤖 Machine Learning Model")

    st.write("""
    A **Random Forest Regression** algorithm is used to predict
    the Mental Health Score.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Training Samples",
            f"{len(X_train):,}"
        )

    with col2:

        st.metric(
            "Testing Samples",
            f"{len(X_test):,}"
        )

    with col3:

        st.metric(
            "Features Used",
            f"{X.shape[1]}"
        )

    st.divider()

    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Mean Absolute Error",
            f"{mae:.2f}"
        )

        st.caption(
            "Lower MAE is generally better."
        )

    with col2:

        st.metric(
            "R² Score",
            f"{r2:.2f}"
        )

        st.caption(
            "Closer to 1 generally indicates better fit."
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("🔍 Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    importance = importance.set_index(
        "Feature"
    )

    st.bar_chart(
        importance["Importance"]
    )

    st.caption(
        "Feature importance indicates how useful each feature was "
        "to the Random Forest model. It does not prove causation."
    )


# ============================================================
# TAB 4 — PREDICTION
# ============================================================

with tab4:

    st.subheader("🎯 Predict Student Mental Health Score")

    st.write(
        "Enter student information below to generate a model-based "
        "Mental Health Score prediction."
    )

    st.info(
        "This prediction is for educational purposes only and "
        "is not a medical or psychological diagnosis."
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "👤 Age",
            min_value=10,
            max_value=100,
            value=20,
            step=1
        )

        gender_options = (
            df["Gender"]
            .dropna()
            .unique()
            .tolist()
        )

        gender = st.selectbox(
            "⚧ Gender",
            gender_options
        )

        education_options = (
            df["Education_Level"]
            .dropna()
            .unique()
            .tolist()
        )

        education = st.selectbox(
            "🎓 Education Level",
            education_options
        )

        social_media = st.number_input(
            "📱 Daily Social Media Hours",
            min_value=0.0,
            max_value=24.0,
            value=float(
                df["Daily_Social_Media_Hours"].median()
            ),
            step=0.1
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        ai_usage = st.number_input(
            "🤖 Daily AI Tool Usage Hours",
            min_value=0.0,
            max_value=24.0,
            value=float(
                df["Daily_AI_Tool_Usage_Hours"].median()
            ),
            step=0.1
        )

        sleep = st.number_input(
            "😴 Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=float(
                df["Sleep_Hours"].median()
            ),
            step=0.1
        )

        physical_activity = st.number_input(
            "🏃 Physical Activity Hours",
            min_value=0.0,
            max_value=24.0,
            value=float(
                df["Physical_Activity_Hours"].median()
            ),
            step=0.1
        )

        physical_health = st.number_input(
            "❤️ Physical Health Score",
            min_value=float(
                df["Physical_Health_Score"].min()
            ),
            max_value=float(
                df["Physical_Health_Score"].max()
            ),
            value=float(
                df["Physical_Health_Score"].median()
            ),
            step=0.1
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Mental Health Score",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [gender],

            "Education_Level": [education],

            "Daily_Social_Media_Hours": [
                social_media
            ],

            "Daily_AI_Tool_Usage_Hours": [
                ai_usage
            ],

            "Sleep_Hours": [
                sleep
            ],

            "Physical_Activity_Hours": [
                physical_activity
            ],

            "Physical_Health_Score": [
                physical_health
            ]
        })


        # Encode categorical values

        input_data = pd.get_dummies(
            input_data,
            columns=[
                "Gender",
                "Education_Level"
            ],
            drop_first=True
        )


        # Match training columns

        input_data = input_data.reindex(
            columns=X.columns,
            fill_value=0
        )


        # Prediction

        prediction = model.predict(
            input_data
        )[0]


        # Keep prediction within dataset range

        min_score = df[
            "Mental_Health_Score"
        ].min()

        max_score = df[
            "Mental_Health_Score"
        ].max()

        prediction = np.clip(
            prediction,
            min_score,
            max_score
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.success(
            f"🧠 Predicted Mental Health Score: "
            f"**{prediction:.2f}**"
        )


        # ----------------------------------------------------
        # SIMPLE INTERPRETATION
        # ----------------------------------------------------

        if prediction < 40:

            interpretation = "Lower predicted score"

        elif prediction < 70:

            interpretation = "Moderate predicted score"

        else:

            interpretation = "Higher predicted score"


        st.info(
            f"📌 Interpretation: **{interpretation}** "
            "based on the model's prediction."
        )

        st.caption(
            "The score interpretation is only a project-level "
            "classification and should not be treated as a "
            "clinical assessment."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">

    **🧠 AI & Social Media Impact**

    Student Health & Mental Health Prediction

    *Machine Learning Project*

    </div>
    """,
    unsafe_allow_html=True
)
