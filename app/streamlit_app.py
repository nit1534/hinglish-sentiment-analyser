import streamlit as st
import tensorflow as tf
import numpy as np

# Page config
st.set_page_config(
    page_title="Hinglish Sentiment Analyser",
    page_icon="🎯",
    layout="centered"
)

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('../models/hinglish_sentiment_model.keras')
    return model

model = load_model()

# Labels
label_map = {0: '😠 Negative', 1: '😊 Positive', 2: '😐 Neutral'}
color_map = {0: '#FF4B4B', 1: '#00C851', 2: '#FFA500'}

# UI
st.title("🎯 Hinglish Sentiment Analyser")
st.markdown("Paste any **Hindi-English mixed text** and get the sentiment instantly.")
st.markdown("---")

# Examples
st.markdown("**Try these examples:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😊 Positive example"):
        st.session_state.input_text = "yaar ye movie ekdum mast thi, loved it so much!"
with col2:
    if st.button("😠 Negative example"):
        st.session_state.input_text = "bilkul bekar service hai, total waste of money"
with col3:
    if st.button("😐 Neutral example"):
        st.session_state.input_text = "kal market mein gaya tha, kuch nahi mila"

# Text input
text_input = st.text_area(
    "Enter Hinglish text:",
    value=st.session_state.get('input_text', ''),
    height=120,
    placeholder="e.g. yaar ye phone ekdum bekar hai, total waste of money..."
)

# Predict
if st.button("Analyse Sentiment", type="primary"):
    if text_input.strip() == '':
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing..."):
            # Predict
            input_tensor = tf.constant([text_input])
            prediction = model.predict(input_tensor, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class] * 100

            # Result
            st.markdown("---")
            st.markdown("### Result")
            
            label = label_map[predicted_class]
            color = color_map[predicted_class]
            
            st.markdown(
                f"<h2 style='color:{color}'>{label}</h2>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Confidence:** {confidence:.1f}%")

            # All 3 scores
            st.markdown("### Confidence breakdown")
            labels = ['Negative', 'Positive', 'Neutral']
            for i, (lab, score) in enumerate(zip(labels, prediction[0])):
                st.progress(float(score), text=f"{lab}: {score*100:.1f}%")

st.markdown("---")
st.markdown(
    "Built with TensorFlow · Trained on SemEval-2020 Hinglish dataset · 92% macro F1"
)
