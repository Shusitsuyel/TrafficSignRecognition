Traffic Sign Recognition App 🚦
Overview
This project is a Traffic Sign Recognition application built using Python, TensorFlow, and Streamlit. It leverages the German Traffic Sign Recognition Benchmark (GTSRB) dataset to train a convolutional neural network (CNN) model that can classify 43 different traffic signs. The app provides two modes for prediction:

Image Upload: Upload a traffic sign image to get a prediction.
Webcam Mode: Use your webcam for real-time traffic sign prediction.

The project includes a full pipeline for data preprocessing, exploratory data analysis (EDA), model training, evaluation, inference, and an interactive Streamlit app.
Features

Preprocesses the GTSRB dataset for training and testing.
Performs EDA to visualize dataset characteristics.
Trains a CNN model with ~97% validation accuracy on 43 traffic sign classes.
Evaluates the model on a test set.
Supports batch inference on test images.
Provides an interactive Streamlit app with:
Image upload for single predictions.
Real-time webcam predictions.
Displays predicted class, class ID, and confidence score.



Dataset
The project uses the GTSRB dataset, which contains:

Training Set: 39,209 images across 43 classes.
Test Set: 12,630 images.
Image Size: Resized to 30x30 pixels for training.
Classes: Includes speed limits, prohibitory signs, warning signs, mandatory signs, and priority signs.

Download the dataset from http://benchmark.ini.rub.de/ and place it in the data/ directory.
Requirements

Python 3.8+
Virtual environment (recommended)

Install the required packages:
pip install -r requirements.txt

Key dependencies include:

tensorflow for model training and inference.
streamlit for the interactive app.
streamlit-webrtc for webcam functionality.
opencv-python for image processing.
numpy, pandas, matplotlib for data handling and visualization.

Project Structure
Trafficsign/
│
├── data/                   # Directory for GTSRB dataset
│   ├── Test/               # Test images
│   ├── Train/              # Training images
│   └── *.csv               # Metadata files
│
├── models/                 # Directory for trained models
│   └── best_model.h5       # Trained CNN model
│
├── outputs/                # Directory for outputs (e.g., inference results)
│
├── scripts/                # Python scripts for the pipeline
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── inference.py
│   └── streamlit_app.py
│
├── main.py                 # Main script to run the pipeline
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation

Setup Instructions

Clone the Repository (after uploading to GitHub):
git clone https://github.com/<your-username>/Trafficsign.git
cd Trafficsign


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download the GTSRB Dataset:

Download from http://benchmark.ini.rub.de/.
Extract and place the Train and Test directories in the data/ folder.


Prepare the Dataset:
python main.py --data



Usage
1. Train the Model
Train the CNN model on the GTSRB dataset:
python main.py --training --epochs 30 --batch_size 32


Model will be saved as models/best_model.h5.

2. Evaluate the Model
Evaluate the trained model on the test set:
python main.py --evaluation

3. Perform Batch Inference
Run inference on all test images:
python main.py --inference --process_all --batch_size 1000


Results are saved in outputs/inference_results.txt.

4. Launch the Streamlit App
Run the interactive app for predictions:
streamlit run main.py -- --streamlit



Example Output

Uploaded Image Prediction:Predicted Class: Vehicles over 3.5t prohibited (ID: 16)
Confidence: 98.75%


Webcam Prediction:Working On


Notes

The model was trained on 30x30 RGB images, so webcam predictions work best with clear, well-lit traffic signs.
If running on OneDrive, pause syncing to avoid file access issues:# Move project to avoid OneDrive issues
mv ~/OneDrive/Desktop/Trafficsign ~/Documents/Trafficsign
cd ~/Documents/Trafficsign



Contributing
Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue to discuss your ideas.
Credits

GTSRB Dataset: Institut für Neuroinformatik, Ruhr-Universität Bochum.
Libraries: TensorFlow, Streamlit, OpenCV, NumPy, Pandas, Matplotlib.
Author: Shushil Suyel.

License
This project is licensed under the MIT License.
