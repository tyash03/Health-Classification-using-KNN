import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("Health Classification using KNN")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Encode categorical columns (e.g., gender)
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    st.subheader("Gender Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='gender', ax=ax1)
    ax1.set_title("Gender Distribution")
    st.pyplot(fig1)

    st.subheader("Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    # Define features and label
    if 'class' not in df.columns:
        st.warning("Your dataset must include a 'class' column as the label.")
    else:
        X = df.drop('class', axis=1)
        y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("KNN Accuracy for Different k Values")
        accuracy_list = []
        k_range = range(1, 21)

        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracy_list.append(acc)

        fig3, ax3 = plt.subplots()
        sns.lineplot(x=list(k_range), y=accuracy_list, ax=ax3)
        ax3.set_xlabel("k value")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("KNN Accuracy for Different k Values")
        st.pyplot(fig3)

        st.subheader("Predict a New Sample")
        gender = st.selectbox("Gender", label_encoders['gender'].classes_)
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        body_fat = st.number_input("Body Fat %", min_value=0.0, max_value=60.0, value=20.0)
        diastolic = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80)
        systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        grip = st.number_input("Grip Force", min_value=0, max_value=100, value=40)
        sitbend = st.number_input("Sit and Bend Forward (cm)", min_value=-20, max_value=50, value=10)
        situps = st.number_input("Sit-ups Count", min_value=0, max_value=100, value=30)
        broadjump = st.number_input("Broad Jump (cm)", min_value=0, max_value=300, value=150)

        if st.button("Predict"):
            input_data = pd.DataFrame([[
                label_encoders['gender'].transform([gender])[0], age, height, weight,
                body_fat, diastolic, systolic, grip, sitbend, situps, broadjump
            ]], columns=X.columns)

            final_model = KNeighborsClassifier(n_neighbors=5)
            final_model.fit(X, y)
            prediction = final_model.predict(input_data)[0]
            st.success(f"Predicted Class: {prediction}")
