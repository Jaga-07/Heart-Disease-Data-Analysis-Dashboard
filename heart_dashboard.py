# heart_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Load Dataset
data = pd.read_csv('heart-1.csv')

# Load Saved Model
model = joblib.load('decision_tree_heart_model.pkl')

# Sidebar Navigation
st.sidebar.title("Heart Disease Dashboard")
section = st.sidebar.radio("Go to", ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Prediction Model"])

st.title("Heart Disease Data Analysis Dashboard")

# Set Target Variable
target = data.columns[-1]

# Univariate Analysis Section
if section == "Univariate Analysis":
    st.header("Univariate Analysis")
    feature = st.selectbox("Select Feature", data.columns[:-1])

    # Histogram
    st.subheader(f"Histogram of {feature}")
    fig, ax = plt.subplots()
    sns.histplot(data[feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot
    st.subheader(f"Boxplot of {feature}")
    fig, ax = plt.subplots()
    sns.boxplot(x=data[feature], ax=ax)
    st.pyplot(fig)

    # Pie Chart for Categorical Features
    if data[feature].nunique() < 10:
        st.subheader(f"Pie Chart of {feature}")
        fig, ax = plt.subplots()
        data[feature].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

# Bivariate Analysis Section
elif section == "Bivariate Analysis":
    st.header("Bivariate Analysis")
    feature = st.selectbox("Select Feature", data.columns[:-1])

    # Scatter Plot
    st.subheader(f"Scatter Plot: {feature} vs {target}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[feature], y=data[target], ax=ax)
    st.pyplot(fig)

    # Bar Plot
    st.subheader(f"Bar Plot: {feature} vs {target}")
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size
    sns.barplot(x=data[feature], y=data[target], ax=ax)
    # Rotate x-axis labels if they are long
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # Add some spacing
    plt.tight_layout()
    st.pyplot(fig)

    # Violin Plot
    st.subheader(f"Violin Plot: {feature} vs {target}")
    fig, ax = plt.subplots()
    sns.violinplot(x=data[target], y=data[feature], ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Multivariate Analysis Section
elif section == "Multivariate Analysis":
    st.header("Multivariate Analysis")

    # Pairplot
    st.subheader("Pair Plot")
    selected_features = st.multiselect("Select Features for Pairplot", data.columns[:-1], default=list(data.columns[:3]))
    if selected_features:
        fig = sns.pairplot(data[selected_features + [target]], hue=target)
        st.pyplot(fig)

    # 3D Visualization
    st.subheader("3D Scatter Plot")
    if len(selected_features) >= 3:
        fig = px.scatter_3d(data, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                            color=target)
        st.plotly_chart(fig)
    else:
        st.info("Please select at least 3 features for 3D plot.")

# Prediction Section
elif section == "Prediction Model":
    st.header("Heart Disease Prediction")

    st.subheader("Enter Patient Details:")
    input_data = []
    for feature in data.columns[:-1]:
        value = st.number_input(f"Enter {feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
        input_data.append(value)

    if st.button("Predict"):
        prediction = model.predict([input_data])[0]
        if prediction == 1:
            st.error("Prediction: Disease Detected")
        else:
            st.success("Prediction: No Disease Detected")

