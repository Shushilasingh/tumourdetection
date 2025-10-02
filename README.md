# tumourdetection
This project implements a multi-class tumour classifier using Keras and TensorFlow. It aims to assist in early diagnosis by accurately identifying tumour types from medical images. The model is trained on labeled tumour images and deployed via a Flask API for real-time predictions.

🚀 Features
Multi-class classification with high validation accuracy

Real-time prediction via Flask API

Confusion matrix and classification report for evaluation

Model saved as .keras for deployment

Mathematical intuition explained cell-by-cell

🏗️ Model Architecture
Framework: Keras (TensorFlow backend)

Loss Function: categorical_crossentropy

Optimizer: adam

Metrics: accuracy

Epochs: 10

📐 Mathematical Insights
🔢 Model Compilation
python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Loss Function: $
𝐿
=
−
∑
𝑖
=
1
𝐶
𝑦
𝑖
log
⁡
(
𝑦
^
𝑖
)
$ Penalizes incorrect predictions based on confidence.

Adam Optimizer: Combines momentum and adaptive learning rates: $
𝜃
𝑡
=
𝜃
𝑡
−
1
−
𝛼
𝑣
^
𝑡
+
𝜖
⋅
𝑚
^
𝑡
$

📈 Model Training
python
history = model.fit(train_generator, epochs=10, validation_data=test_generator)
Forward Pass: $
𝑎
(
𝑙
)
=
𝑓
(
𝑊
(
𝑙
)
𝑎
(
𝑙
−
1
)
+
𝑏
(
𝑙
)
)
$

Backpropagation: $
∂
𝐿
∂
𝑊
(
𝑙
)
=
∂
𝐿
∂
𝑎
(
𝑙
)
⋅
∂
𝑎
(
𝑙
)
∂
𝑊
(
𝑙
)
$

Accuracy: $
Accuracy
=
Correct Predictions
Total Predictions
$

💾 Model Saving
python
model.save('tumour_classifier_model.keras')
Serializes weights 
𝑊
(
𝑙
)
, biases 
𝑏
(
𝑙
)
, and architecture.

🧠 Prediction
python
preds = model.predict(test_generator)
predicted_classes = np.argmax(preds, axis=1)
Softmax Activation: $
𝑦
^
𝑖
=
𝑒
𝑧
𝑖
∑
𝑗
=
1
𝐶
𝑒
𝑧
𝑗
$

Argmax: $
Predicted Class
=
arg
⁡
max
⁡
𝑖
𝑦
^
𝑖
$

📊 Evaluation
python
cm = confusion_matrix(true_classes, predicted_classes)
classification_report(true_classes, predicted_classes, target_names=class_labels)
Confusion Matrtix

📊 Performance Snapshot
Epoch	Train Accuracy	Val Accuracy	Val Loss
1	52.8%	78.3%	0.5570
5	92.3%	90.5%	0.2603
10	95.9%	94.4%	0.1723
💻 Deployment
Flask API wraps the trained model

Accepts image uploads and returns tumour class

Frontend in progress for user-friendly interface

📁 File Structure
Code
tumourdetection/
├── tumour_classifier_model.keras
├── app.py
├── templates/
├── static/
├── README.md
└── ...
🎯 Why This Matters
Early tumour detection can save lives. This project demonstrates how deep learning can be applied to real-world health diagnostics, especially in resource-constrained settings. The goal is to make AI-powered screening accessible, fast, and reliable.
