🎙️ Speech Emotion Recognition
📌 Project Overview

Speech Emotion Recognition (SER) is a machine learning and deep learning–based project that focuses on identifying human emotions from speech signals. Speech carries not only linguistic information but also emotional cues such as mood, stress, and intent. Recognizing these emotions plays a crucial role in building intelligent and human-like systems.

This project explores traditional machine learning, deep neural networks, and sequence-based deep learning models to analyze and classify emotions from speech audio data.

🎯 Objectives

Extract meaningful acoustic features from speech signals

Compare traditional ML, DNN, and sequence-based DL models

Accurately classify emotions from audio recordings

Evaluate model performance using accuracy and F1-score

🧠 Emotions Recognized

The system recognizes the following eight emotional classes:

Neutral

Calm

Happy

Sad

Angry

Fearful

Disgust

Surprised

🛠️ Technologies & Tools

Programming Language: Python

Libraries & Frameworks:

NumPy

Pandas

Librosa

Scikit-learn

TensorFlow / Keras

Matplotlib

Models Used:

XGBoost

Fully Connected Deep Neural Network (DNN)

GRU (Gated Recurrent Unit)

LSTM (Long Short-Term Memory)

Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

⚙️ Methodology
1️⃣ Preparation

Audio files converted to a unified format and sampling rate

Silence removal and amplitude normalization applied

Noise reduction to improve feature quality

2️⃣ Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) extracted

Features represent emotional characteristics of speech

Feature scaling applied using standardization

3️⃣ Traditional Machine Learning Models

Classical ML models trained using extracted features

Ensemble-based models (e.g., XGBoost) used as baselines

4️⃣ Deep Neural Networks (DNN)

Fully connected neural networks implemented

Dense layers with ReLU activation used

Improved feature representation learning

5️⃣ Sequence-Based Deep Learning Models

LSTM and GRU models employed

Temporal dependencies captured from sequential speech data

Better modeling of emotional variations over time

6️⃣ Evaluation

Dataset split into 80% training and 20% testing

Evaluation metrics: Accuracy and F1-score

Fair comparison across all models

📊 Dataset Description

The RAVDESS dataset consists of high-quality speech recordings collected in controlled studio conditions.

24 professional actors (male & female)

WAV audio format

Average duration: ~3 seconds

Balanced emotional classes

This dataset is widely used for benchmarking Speech Emotion Recognition systems.

📈 Results (Comparison of All Approaches)
Approach	Model	Accuracy	F1-Score
Feature Engineering + ML	XGBoost	94%	94%
Fully Connected DNN	DNN (ReLU)	88%	88%
Sequence-Based Model	GRU	93.75%	93.75%
🧪 Discussion

Traditional ML Models:
XGBoost achieved the highest accuracy among classical models due to effective feature engineering. Linear models performed poorly because speech emotions exhibit non-linear patterns.

Deep Neural Networks:
DNNs improved performance compared to basic ML models but failed to fully capture temporal dependencies in speech signals.

Sequence-Based Models:
GRU and LSTM models performed best overall. GRU achieved high accuracy with lower computational cost compared to LSTM, making it the most suitable model for SER tasks.

✅ Conclusion

This project demonstrates that:

Ensemble-based ML models provide strong baseline performance

Deep neural networks improve representation learning

Sequence-based models, especially GRU, are the most effective for Speech Emotion Recognition due to their ability to capture temporal information

🔮 Future Work

Combine feature engineering with sequence-based deep learning

Explore attention mechanisms and transformer-based models

Extend dataset with more languages and real-world speech

Deploy SER system in real-time applications

👩‍💻 Author

Rabia Sadiq

📜 License

This project is for academic and educational purposes.
