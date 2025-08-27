import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, confusion_matrix, classification_report, ConfusionMatrixDisplay

st.title('💲 Approx. Total Revenue App')
st.info('This is app builds a machine learning model to predict approximation total revenue from dairy product using Random Forest Classification.')

with st.expander('Data Overview'):
    df = pd.read_csv('https://raw.githubusercontent.com/ggyymmnastiar/g-revenue/refs/heads/master/dairy_dataset.csv')
    
    st.write(f"**● Total Samples:** {len(df)}")
    
    st.write('**● Raw Data:**')
    df

    st.write('**● X raw (Input):**')
    x_raw = df.drop('Approx. Total Revenue(INR)', axis=1)
    x_raw

# Convert INR to IDR
df['Price per Unit (sold) (IDR)'] = df['Price per Unit (sold)'] * 185
df['Approx. Total Revenue (IDR)'] = df['Approx. Total Revenue(INR)'] * 185

with st.expander('Data Preparation & EDA'):
    st.write('**● Convert column from Rupee (INR) to Rupiah (IDR)**')
    st.write('    Assuming an exchange rate of 1 INR = 185 IDR')
    st.code(""" 
    df['Price per Unit (sold) (IDR)'] = df['Price per Unit (sold)'] * 185
    df['Approx. Total Revenue (IDR)'] = df['Approx. Total Revenue(INR)'] * 185
    """, language='python')

    # Check missing value
    missing_counts = df.isnull().sum()
    st.write("**● Missing Values per Column:**")
    st.write(missing_counts)

    # Define X and y
    st.write('**● Define X and y**')
    X = df[['Number of Cows', 'Farm Size', 'Product Name', 'Quantity (liters/kg)', 
                 'Shelf Life (days)', 'Storage Condition', 'Quantity Sold (liters/kg)', 
                 'Sales Channel', 'Price per Unit (sold) (IDR)']]
    y = df['Approx. Total Revenue (IDR)']

    st.code("""
    X = df[['Number of Cows', 'Farm Size', 'Product Name', 'Quantity (liters/kg)', 
                 'Shelf Life (days)', 'Storage Condition', 'Quantity Sold (liters/kg)', 
                 'Sales Channel', 'Price per Unit (sold) (IDR)']]
    y = df['Approx. Total Revenue (IDR)']
    """, language='python')

