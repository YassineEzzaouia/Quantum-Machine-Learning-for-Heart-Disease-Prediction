This repository contains a hybrid quantum-classical machine learning project for predicting heart disease using the UCI Heart Disease dataset.

--------------------------------------🎯 Main Objectives-------------------------------------

This project titled Quantum Machine Learning for Heart Disease Prediction aims to:

-Predict heart disease presence using the Cleveland Heart Disease dataset.

-The target variable (num) indicates the presence (1–4) or absence (0) of heart disease.

-For modeling, the problem is treated as binary classification:

    0 → No heart disease

    1 → Heart disease present

-Compare classical and quantum machine learning models:

    Evaluate performance differences between traditional models (e.g., Random Forest, SVM) and a hybrid quantum-classical model.

    Perform automatic hyperparameter tuning using Optuna for better model optimization.

    Visualize and interpret results through metrics such as accuracy, ROC curves, and AUC to assess and compare model performance.

--------------------------------------🧠 Tools and Technologies Used--------------------------------------
--------------------------------------🧩 Core Libraries--------------------------------------

-Pandas, NumPy → Data manipulation and preprocessing

-Matplotlib, Seaborn → Data visualization

-JSON → Result export and storage

--------------------------------------⚙️ Machine Learning (Classical)--------------------------------------

-Scikit-learn (sklearn)

-Model training (LogisticRegression, RandomForestClassifier, SVC)

-Data splitting (train_test_split), scaling (StandardScaler), and dimensionality reduction (PCA)

-Evaluation metrics (accuracy_score, roc_auc_score, roc_curve, auc)

-XGBoost → Gradient boosting classification model

-Optuna → Automatic hyperparameter optimization

--------------------------------------🧬 Quantum Machine Learning--------------------------------------

-PennyLane (qml) → Framework for creating and training variational quantum circuits (VQCs)

-Used to encode features and perform optimization in a hybrid quantum-classical architecture.
