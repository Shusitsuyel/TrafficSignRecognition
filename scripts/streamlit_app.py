import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import streamlit as st
import argparse
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Class names for GTSRB dataset (corrected No Parking label)
CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)", 3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)", 6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)", 9: "No parking", 10: "No passing for vehicles over 3.5t", 11: "Right-of-way at the next intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5t prohibited", 17: "No entry",
    18: "General caution", 19: "Dangerous curve to the left", 20: "Dangerous curve to the right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right", 25: "Road work", 26: "Traffic signals",
    27: "Pedestrians", 28: "Children crossing", 29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only",
    36: "Go straight or right", 37: "Go straight or left", 38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing by vehicles over 3.5t"
}

# Load the model (cached to avoid reloading)
@st.cache_resource
def load_traffic_sign_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image for prediction
def preprocess_image(image):
    if image.shape[-1] == 4:  # Convert RGBA to RGB if needed
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[-1] == 1:  # Convert grayscale to RGB if needed
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Enhance contrast for better detection
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)  # Increase contrast
    image = cv2.resize(image, (30, 30), interpolation=cv2.INTER_NEAREST)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Shape: (1, 30, 30, 3)
    return image

# Predict traffic sign
def predict_traffic_sign(model, image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100
    predicted_class_name = CLASS_NAMES.get(predicted_class, "Unknown")
    # Debug: Print raw prediction scores
    print(f"Raw prediction scores: {prediction[0]}")
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
    return predicted_class_name, predicted_class, confidence, image

# Video processor for webcam
class TrafficSignVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.predicted_class_name = None
        self.predicted_class = None
        self.confidence = None
        self.frame = None
        self.min_confidence = 50.0  # Minimum confidence threshold

    def recv(self, frame):
        # Convert WebRTC frame to OpenCV format
        img = frame.to_ndarray(format="bgr")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Predict traffic sign
        predicted_class_name, predicted_class, confidence, self.frame = predict_traffic_sign(self.model, img_rgb)

        # Update predictions only if confidence is above threshold
        if confidence >= self.min_confidence:
            self.predicted_class_name = predicted_class_name
            self.predicted_class = predicted_class
            self.confidence = confidence
        else:
            self.predicted_class_name = None
            self.predicted_class = None
            self.confidence = None

        return frame

def run(args):
    model_dir = args.model_dir

    print("----------- Starting Streamlit App -----------")
    print("Loading trained model...")
    
    # Load the trained model
    model_path = os.path.join(model_dir, 'best_model.h5')
    model = load_traffic_sign_model(model_path)
    if model is None:
        return

    # Streamlit app
    st.title("🚦 Traffic Sign Recognition App")
    st.markdown("""
        Welcome to the Traffic Sign Recognition App! You can either upload an image or use your webcam to predict traffic signs in real-time.
        The model was trained on the GTSRB dataset and can recognize 43 different traffic signs.
    """)

    # Mode selection
    mode = st.radio("Select Mode", ["Upload Image", "Webcam"])

    if mode == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "ppm"])
        
        if uploaded_file is not None:
            try:
                # Load and preprocess the image
                print("Loading and preprocessing uploaded image...")
                image = Image.open(uploaded_file)
                image = np.array(image)
                predicted_class_name, predicted_class, confidence, display_image = predict_traffic_sign(model, image)

                # Display results
                st.subheader("Prediction Result")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(display_image, caption="Uploaded Image", use_container_width=True)  # Updated parameter
                with col2:
                    if confidence >= 50.0:
                        st.write(f"**Predicted Class**: {predicted_class_name} (ID: {predicted_class})")
                        st.write(f"**Confidence**: {confidence:.2f}%")
                    else:
                        st.write("Confidence too low to make a prediction. Try a clearer image.")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    else:  # Webcam mode
        st.subheader("Webcam Live Prediction")
        st.write("Click 'Start' to begin capturing video from your webcam. Hold a traffic sign in front of the camera to get a prediction.")
        # st.write("**Tip**: For hand-drawn signs, use bright colors (e.g., red circle, blue center for 'No Parking'), ensure good lighting, and hold the sign close to the camera.")

        # WebRTC configuration
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        # Start webcam streaming
        webrtc_ctx = webrtc_streamer(
            key="traffic-sign-recognition",
            video_processor_factory=lambda: TrafficSignVideoProcessor(model),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
        )

        if webrtc_ctx.video_processor:
            processor = webrtc_ctx.video_processor
            if processor.frame is not None:
                # Display the latest frame and prediction
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(processor.frame, caption="Live Webcam Feed", use_container_width=True)  # Updated parameter
                with col2:
                    if processor.predicted_class_name:
                        st.write(f"**Predicted Class**: {processor.predicted_class_name} (ID: {processor.predicted_class})")
                        st.write(f"**Confidence**: {processor.confidence:.2f}%")
                    else:
                        st.write("No prediction yet or confidence too low. Hold a clear traffic sign in front of the camera.")

    print("----------- Streamlit App Running -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit App for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    args = parser.parse_args()
    run(args)