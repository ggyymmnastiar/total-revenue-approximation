import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, confusion_matrix, classification_report, ConfusionMatrixDisplay

st.title('💲 Approx. Total Revenue App')
st.info('This is app builds a machine learning model to predict approximation total revenue from dairy product using Random Forest Regression.')

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
    st.write("**● Missing Values per Column**")
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

    # Final Dataset
    st.write('**● Final Dataset**')
    df_final = pd.concat([X, y], axis=1)
    df_final

    # Define numerical & categorical column
    st.write('**● Define Numerical & Categorical Column**')
    numerical_cols = ['Number of Cows', 'Quantity (liters/kg)', 'Shelf Life (days)',
                  'Quantity Sold (liters/kg)', 'Price per Unit (sold) (IDR)']
    categorical_cols = ['Farm Size', 'Product Name', 'Storage Condition', 'Sales Channel']

    st.code("""
    numerical_cols = ['Number of Cows', 'Quantity (liters/kg)', 'Shelf Life (days)',
                  'Quantity Sold (liters/kg)', 'Price per Unit (sold) (IDR)']
    categorical_cols = ['Farm Size', 'Product Name', 'Storage Condition', 'Sales Channel']
    """, language='python')

# Data Visualization
with st.expander('Data Visualization'):
    st.write('**SOON**')

# Input Features (Sidebar)
with st.sidebar:
    st.header('Input Features')

    number_of_cows = st.slider('Number of Cows', min_value=1, max_value=150, value=50)
    shelf_life = st.slider('Shelf Life (days)', min_value=1, max_value=60, value=15)

    farm_size = st.selectbox('Farm Size', option=['Large', 'Medium', 'Small'])
    product_name = st.selectbox('Product Name', option=['Milk', 'Cheese', 'Yoghurt'])
    storage_condition = st.selectbox('Storage Condition', option=['Refrigerated', 'Frozen', 'Ambient', 'Polythene Packet', 'Tetra Pack'])
    sales_channel = st.selectbox('Sales Channel', option=['Retail', 'Wholesale', 'Online'])

    quantity_total = st.number_input(
        'Quantity in Stock (liters/kg)', 
        min_value=0, 
        value=500, 
        step=10, 
        format="%d"
    )
    quantity_sold = st.number_input(
        'Quantity Sold (liters/kg)', 
        min_value=0, 
        value=450, 
        step=10, 
        format="%d"
    )
    price_per_unit = st.number_input(
        'Price per Unit (sold)', 
        min_value=0, 
        value=25000, 
        step=500, 
        format="%d"
    )



# Input Features
with st.expander('Input Features'):
    st.write('**SOON**')

with st.expander('Data Preprocessing'):
    tab1, tab2, tab3 = st.tabs(["Pipeline Setup", "Train | Test", "Hyperparameter Tuning"])
    
    with tab1:
        st.subheader("Preprocessing Pipeline")

        # Feature Groups
        st.write("**Feature Groups**")
        st.code("""
        numerical_cols = ['Number of Cows', 'Quantity (liters/kg)', 'Shelf Life (days)',
                    'Quantity Sold (liters/kg)', 'Price per Unit (sold) (IDR)']
        categorical_cols = ['Farm Size', 'Product Name', 'Storage Condition', 'Sales Channel']
        """, language='python')

        # Preprocessor & Pipeline
        st.write("**Define Preprocessor and Pipeline**")
        preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ])

        pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", RandomForestRegressor())])
        
        st.code("""
        preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ])

        pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", RandomForestRegressor())])
        """, language='python')

    with tab2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                           test_size=0.2, random_state=101)
        st.subheader("Train | Test Split")
        st.write(f"**Train Size:** {len(X_train)} samples")
        st.write(f"**Test Size:** {len(X_test)} samples")
        st.write(f"**Random State:** 101")
        st.write(f"**Test Split:** 20%")

        st.code("""
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                           test_size=0.2, random_state=101)
        """, language='python')

        # Train Data
        st.subheader("Train Data")

        st.code("""
        pipeline.fit(X_train, y_train)
        """, language='python')

        if st.button("Train the Data (.fit)"):
            with st.spinner('Model is being trained, please wait...'):
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
            st.success('Model training completed!')

            st.subheader("Model Evaluation")
            # 1. Hitung Metrik
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            # 2. Tampilkan Metrik menggunakan st.metric dalam kolom
            # --- Baris Pertama ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="R-squared (R²)", 
                        value=f"{r2:.3f}", 
                        help="Semakin mendekati 1, semakin baik modelnya.")
            with col2:
                st.metric(label="Mean Absolute Percentage Error (MAPE)", 
                        value=f"{mape:.2f} %",
                        help="Rata-rata persentase kesalahan. Semakin kecil, semakin baik.")

            # --- Baris Kedua ---
            col3, col4 = st.columns(2)
            with col3:
                st.metric(label="Mean Absolute Error (MAE)", 
                        value=f"Rp {mae:,.0f}",
                        help="Rata-rata selisih absolut antara prediksi dan nilai asli dalam Rupiah.")
            with col4:
                st.metric(label="Root Mean Squared Error (RMSE)", 
                        value=f"Rp {rmse:,.0f}",
                        help="Akar kuadrat dari rata-rata kesalahan kuadrat. Memberi bobot lebih pada kesalahan besar.")
            
            # --- Tampilkan Visualisasi ---
            st.subheader("Result Visualizations")

            # Grafik 1: Actual vs. Predicted
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
            ax.set_xlabel("Actual Revenue (IDR)")
            ax.set_ylabel("Predicted Revenue (IDR)")
            ax.set_title("Actual vs Predicted Revenue")
            st.pyplot(fig)
            st.write("Grafik di atas membandingkan nilai pendapatan aktual dengan hasil prediksi. Semakin dekat titik-titik ke garis merah putus-putus, semakin akurat prediksinya.")

    with tab3:
        st.subheader("Hyperparameter Tuning")