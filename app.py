import streamlit as st
import requests
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Manufacturing QC Assistant",
    page_icon="🏭",
    layout="wide"
)

# --- API Endpoints ---
# Make sure your FastAPI backend is running at this address
BACKEND_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
ASK_ENDPOINT = f"{BACKEND_URL}/ask"


# --- Main App Interface ---
st.title("🏭 Manufacturing Quality Control Assistant")
st.write("""
This application uses AI to assist with quality control. 
You can upload an image of a cast part to check for defects, 
and ask questions about the quality control history.
""")

st.divider()

# --- Two-Column Layout ---
col1, col2 = st.columns(2)

# --- Column 1: Defect Detection ---
with col1:
    st.header("1. Defect Detection")
    st.write("Upload an image of a cast part to classify it as 'defective' or 'ok'.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        # When the user clicks the button, send the image to the backend
        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                # Convert image to bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                # Send request to the backend
                files = {'file': (uploaded_file.name, img_bytes, uploaded_file.type)}
                try:
                    response = requests.post(PREDICT_ENDPOINT, files=files)
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['prediction']
                        confidence = result['confidence']
                        
                        if prediction == 'ok_front':
                            st.success(f"Prediction: OK ({confidence:.2%} confidence)")
                        else:
                            st.error(f"Prediction: DEFECTIVE ({confidence:.2%} confidence)")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the backend: {e}")

# --- Column 2: Quality Control Q&A ---
with col2:
    st.header("2. Quality Control Q&A")
    st.write("Ask a question about the prediction history (e.g., 'How many defects were found today?').")

    user_question = st.text_input("Your question:")

    if st.button('Ask Gemini'):
        if user_question:
            with st.spinner('Thinking...'):
                try:
                    response = requests.post(ASK_ENDPOINT, json={"text": user_question})
                    if response.status_code == 200:
                        result = response.json()
                        st.info("Answer:")
                        st.write(result['answer'])
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the backend: {e}")
        else:
            st.warning("Please enter a question.")