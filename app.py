import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import math

# --- CONFIGURATION (MUST MATCH TRAINING) ---
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # Training duration
NUM_SEGMENTS = 5    # Segments per track
HOP_LENGTH = 512
N_MFCC = 13
N_FFT = 2048

# Calculate the exact number of samples per segment expected by the model
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

# Genre Labels (Order is critical: must match your training folder order 0-9)
GENRES = ["Blues", "Classical", "Country", "Disco", "Hiphop", 
          "Jazz", "Metal", "Pop", "Reggae", "Rock"]

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Load the trained model once
    model = tf.keras.models.load_model("music_genre_cnn.keras")
    return model

model = load_model()

# --- 2. PREPROCESSING FUNCTION ---
def process_input_file(uploaded_file):
    """
    Takes a random audio file, slices it into 3-second chunks,
    and returns a batch of MFCCs ready for the model.
    """
    # Load the audio file (librosa handles decoding)
    # sr=22050 ensures consistency with training
    y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
    
    # Calculate how many full 3-second segments fit in this file
    num_segments = int(len(y) / SAMPLES_PER_SEGMENT)
    
    print(f"Audio length: {len(y)/sr:.2f}s | Segments extracted: {num_segments}")
    
    batch_mfcc = []
    
    for i in range(num_segments):
        start = SAMPLES_PER_SEGMENT * i
        finish = start + SAMPLES_PER_SEGMENT
        
        # Extract MFCC for this slice
        mfcc = librosa.feature.mfcc(y=y[start:finish], sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T # Transpose to match (Time, Features)
        
        # Exact shape check: The model expects specific dimensions (e.g., 130x13)
        expected_vectors = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)
        
        if len(mfcc) == expected_vectors:
            batch_mfcc.append(mfcc.tolist())
            
    return np.array(batch_mfcc)

# --- 3. FRONTEND UI ---
st.title("🎵 AI Music Genre Detector")
st.markdown("Upload any **MP3** or **WAV** file. The AI will listen to the track and detect its genre.")

# File Uploader
uploaded_file = st.file_uploader("Drop a song here", type=["mp3", "wav"])

if uploaded_file is not None:
    # Show audio player
    st.audio(uploaded_file, format="audio/mp3")
    
    if st.button("Analyze Genre"):
        with st.spinner("Listening and analyzing..."):
            try:
                # A. Preprocess the audio into a batch of segments
                X_batch = process_input_file(uploaded_file)
                
                # Reshape for CNN input: (Batch_Size, Time, MFCC, Channels)
                # We add the '1' at the end for the grayscale channel
                X_batch = X_batch[..., np.newaxis]
                
                # B. Predict
                # This returns a probability list for EVERY segment
                # Shape: (Num_Segments, 10_Genres)
                predictions = model.predict(X_batch)
                
                # C. Aggregation (The Voting Logic)
                # We average the probabilities across all segments
                avg_preds = np.mean(predictions, axis=0)
                
                # D. Final Result
                top_genre_index = np.argmax(avg_preds)
                top_genre = GENRES[top_genre_index]
                confidence = avg_preds[top_genre_index]
                
                # E. Display
                st.balloons()
                st.success(f"Prediction: **{top_genre}**")
                st.metric("Confidence Score", f"{confidence * 100:.1f}%")
                
                # Detailed Breakdown Chart
                st.write("---")
                st.write("### Genre Probability Breakdown")
                st.bar_chart(dict(zip(GENRES, avg_preds)))

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Note: The file might be too short. Try a song longer than 5 seconds.")