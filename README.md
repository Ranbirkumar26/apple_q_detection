website link --> https://apple-q-detection-2.onrender.com/


 Apple Quality Prediction

This project is a **machine learning + web app** pipeline to predict the quality of apples (good or bad) based on their physical and chemical features.  
It uses **Support Vector Classifier (SVC)** with hyperparameter tuning, data preprocessing (scaling + SMOTE), and exposes a **Flask web interface** for real-time predictions.

---

 Project Structure


.
├── app.py                   Flask web application
├── train.py                 Training script for ML model
├── data.csv                 Dataset (input for training)
├── models/                  Saved model, scaler, and metrics
│   ├── apple\_quality\_model.pkl
│   ├── scaler.pkl
│   └── model\_metrics.pkl
├── templates/
│   └── index.html           Frontend HTML for input form and results
└── README.md                Project documentation

`

---

 Features
- Data Preprocessing
  - Removes unused columns (`A_id`)
  - Handles missing values
  - Scales data using `RobustScaler`, `StandardScaler`, and `MinMaxScaler`
  - Balances dataset with **SMOTE**
- Model Training
  - Uses **Support Vector Classifier (SVC)** with `RandomizedSearchCV` for hyperparameter tuning
  - Evaluates model with accuracy, confusion matrix, and classification report
- Deployment
  - Flask app for interactive predictions
  - Takes **7 input features** from user and predicts if the apple is **good** or **bad**
  - Displays **prediction with confidence score**

---

 Dataset
The dataset (`data.csv`) should contain the following columns:

| Feature       | Description                  |
|---------------|------------------------------|
| `Size`        | Size of the apple            |
| `Weight`      | Weight of the apple          |
| `Sweetness`   | Sweetness level              |
| `Crunchiness` | Crunchiness level            |
| `Juiciness`   | Juiciness level              |
| `Ripeness`    | Ripeness score               |
| `Acidity`     | Acidity level                |
| `Quality`     | Target column (`good`/`bad`) |

---

 Usage

 1. Install Dependencies
bash
pip install -r requirements.txt
`

Example `requirements.txt`:

txt
pandas
numpy
scikit-learn
imblearn
flask
joblib


---

 2. Train the Model

bash
python train.py


This will:

* Preprocess data
* Train the best SVC model
* Save the model, scaler, and metrics in the `models/` folder

---

 3. Run the Web App

bash
python app.py


The app will start on:


http://127.0.0.1:5000/


---

 Web App Workflow

1. Open the app in your browser
2. Enter values for:

   * Size
   * Weight
   * Sweetness
   * Crunchiness
   * Juiciness
   * Ripeness
   * Acidity
3. Click **Predict**
4. Get:

   * Prediction: **Good** / **Bad**
   * Confidence score (e.g., 92.35%)

---

 Model Performance

The training script saves performance metrics in:


models/model_metrics.pkl


It includes:

* Accuracy
* Confusion Matrix
* Classification Report
* Best Hyperparameters

---

 Tech Stack

* Python
* scikit-learn
* imbalanced-learn (SMOTE)
* Flask
* Joblib
* HTML (Flask templates)

---

 Future Improvements

* Add more features for better accuracy
* Try other ML models (Random Forest, XGBoost, etc.)
* Deploy on cloud platforms (Heroku, Render, AWS)
* Create a better frontend (Bootstrap or React)

---

 Author

Developed by **Ranbir Kumar**
For academic and practical use



