import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import streamlit as st
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from PIL import Image
import os

# Streamlit app title
st.title('Program Prediksi Kepuasan Hidup Penumpang KRL di Indonesia dengan Menggunakan Algoritma C4.5')

# Create tabs
tab1, tab2 = st.tabs(["Dataset & Model", "Prediksi Kepuasan Hidup"])

# Load data
data = pd.read_csv('data.csv', delimiter=';')

with tab1:
    # Display data in Streamlit app
    st.subheader('Dataset')
    st.dataframe(data)

    # Preprocess data
    data['Kepuasan Hidup'] = data['Kepuasan Hidup'].map({'Sangat Tidak Puas': 0, 'Sangat Puas': 1})

    # Split data
    X = data.drop('Kepuasan Hidup', axis=1)
    y = data['Kepuasan Hidup']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    st.subheader('Hasil perhitungan rata-rata Gain')
    st.text(report)

    # Visualize and save the decision tree as a JPEG file
    fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the size as needed
    plot_tree(model, filled=True, ax=ax, feature_names=X.columns, class_names=['Sangat Tidak Puas', 'Sangat Puas'])
    plt.savefig('decision_tree.jpg')
    plt.close()

    # Display the decision tree image
    st.subheader('Decision Tree')
    image = Image.open('decision_tree.jpg')
    st.image(image, caption='Decision Tree')

    # Function to download JPEG file in Streamlit
    def get_image_download_link(img_path):
        with open(img_path, "rb") as file:
            btn = st.download_button(
                label="Download Decision Tree as JPEG",
                data=file,
                file_name="decision_tree.jpg",
                mime="image/jpeg"
            )
        return btn

    # Provide download link for the JPEG file
    get_image_download_link('decision_tree.jpg')

    # Optional: Clean up by removing the image file if not needed anymore
    os.remove('decision_tree.jpg')

with tab2:
    # Sidebar for predictions
    st.subheader('Prediksi Kepuasan Hidup')
    input_features = [st.number_input(feature, min_value=0) for feature in X.columns]
    if st.button('Predict'):
        prediction = model.predict([input_features])[0]
        st.write('Predicted Class:', 'Sangat Puas' if prediction == 1 else 'Sangat Tidak Puas')

# Streamlit app footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #212422;
        color: #8c968f;
        text-align: center;
        padding: 5px;
    }
    </style>
    <div class="footer">
        <p>Developed by Diva Reihan Ferdian Utomo, Ibadurrohman Al Aufa, Sofyan Nur Rohman.</p>
    </div>
    """, unsafe_allow_html=True)