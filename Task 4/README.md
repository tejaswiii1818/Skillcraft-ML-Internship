# ✋ Hand Gesture Recognition using SVM

This project is part of the **SCT_ML_02** task, aimed at building a machine learning model that can recognize and classify static hand gestures using Support Vector Machines (SVM). The system can be used for intuitive human-computer interaction, touchless interfaces, or assistive technologies.

---

## 📁 Dataset

We used the [leapGestRecog Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) which contains 10 gesture classes performed by 10 different users.

Each sample is a grayscale image of size 320x240.

**Gesture classes include:**
- Palm  
- Fist  
- Thumb  
- Index  
- OK  
- L  
- C  
- Down  
- Fist Moved  
- Palm Moved

---

## 🔧 Preprocessing

- All images were resized to `64x64`.
- Normalized pixel values to the `[0, 1]` range.
- Flattened for SVM input.
- Selected a subset (`max_per_class = 50`) for faster training.

---

## 🤖 Model

- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Cross-Validation:** Stratified 5-fold CV
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report

---

## 📊 Results

| Metric     | Value |
|------------|-------|
| Accuracy   | ~94%  |
| Precision  | High  |
| Recall     | High  |

> The model performed well across all gesture classes with minimal confusion between similar gestures.

---

## Project Structure

SCT_ML_02/
├── dataset/
│   └── leapGestRecog/
├── utils/
│   └── data_utils.py
├── gesture_preprocessing.ipynb
├── model_training.ipynb
├── README.md
└── requirements.txt
 
 ---

##👤 Author
Ayush Kumar Rai
3rd Year CSE Student – Sai Vidya Institute of Technology
GitHub: @ayu-yishu13

 ---

##📌 Future Work
Real-time gesture detection using webcam.

Deep learning implementation with CNNs.

Web or mobile app integration.

 ---

##🌟 Acknowledgements
leapGestRecog Dataset - Kaggle

Scikit-learn, OpenCV, NumPy

 ---
![Screenshot (50)](https://github.com/user-attachments/assets/b514b93e-1e38-47b5-a625-06b61e27486f)
![Screenshot (52)](https://github.com/user-attachments/assets/d3c847c2-3ec0-4271-9546-a21dabb3e365)
![Screenshot (48)](https://github.com/user-attachments/assets/986840e5-779c-4a8d-b705-872ebd56678e)
![Screenshot (49)](https://github.com/user-attachments/assets/6740218c-9185-481a-8539-69377cf10dfb)


##🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/ayu-yishu13/SCT_ML_02.git
cd SCT_ML_02

# Install dependencies
pip install -r requirements.txt![Screenshot (52)](https://github.com/user-attachments/assets/43f38512-c1d5-4e3a-9253-24759fd69e70)


# Run notebook
jupyter notebook
