\# Automated Diagnosis of Diabetic Retinopathy using Deep Learning



I concluded that a custom Ordinal Regression approach outperforms traditional Softmax classification, delivering superior clinical performance. By adapting an EfficientNet-B3 backbone with an ordinal inference logic and Focal Loss, the final model significantly outperformed baseline configurations, reducing critical false-negative diagnoses by 35%. 



Supported by advanced OpenCV preprocessing (CLAHE and Ben Graham's transform with circular masking), the system achieved a 0.685 Quadratic Weighted Kappa (QWK) score on a highly imbalanced test set of over 53,000 images. The final solution is available as a low-latency, interactive web application using Streamlit.



\## 🛠️ Tech Stack

\- \*\*Deep Learning:\*\* TensorFlow / Keras (Ordinal Regression, Focal Loss, Cosine Warmup Scheduler)

\- \*\*Computer Vision:\*\* OpenCV, Albumentations (CLAHE, Ben Graham, Circular Masking, Auto-Crop)

\- \*\*Data Science:\*\* Python, NumPy, Pandas, Scikit-Learn

\- \*\*Deployment:\*\* Streamlit



\## 🚀 How to run the Streamlit Demo



1\. Clone this repository:

`git clone https://github.com/Username/Diabetic-Retinopathy-Deep-Learning.git`



2\. Install dependencies:

`pip install -r requirements.txt`



3\. Run the app:

`streamlit run app.py`

