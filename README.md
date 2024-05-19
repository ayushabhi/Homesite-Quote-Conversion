# Homesite-Quote-Conversion
Predict the likelihood of a customer purchasing an insurance plan in this Kaggle competition. Explore innovative techniques like SMOTE for imbalanced data and ensemble predictions through stacking to refine your machine learning skills and enhance predictive modeling expertise.

# Introduction
In this advanced machine learning assignment, we delve into innovative techniques for addressing real-world challenges. The focus is on handling imbalanced data using SMOTE and implementing ensemble predictions through stacking. As we participate in the Homesite Quote Conversion Kaggle competition, we will harness the power of various classification methods and navigate preprocessed datasets. Join us on this journey to refine your machine learning skills and elevate your expertise in predictive modeling.

# Objective
1. **Apply SMOTE:** Use the Synthetic Minority Over-sampling Technique (SMOTE) and its variations to handle imbalanced data, improving accuracy in predicting the minority class.
2. **Utilize Stacking:** Combine predictions from diverse classifiers, including Decision Trees, Random Forest, Support Vector Machines, Multilayer Perceptron, and K-Nearest Neighbors, through stacking techniques to create a more robust predictive model.
3. **Engage in the Kaggle Competition:** Participate in the Homesite Quote Conversion Kaggle competition, leveraging preprocessed datasets and Scikit-Learn classification methods to predict the probability of customer insurance plan purchases.
4. **Experiment with SMOTE:** Test varying percentages of SMOTE to enhance accuracy in predicting the minority class, focusing on effective handling of imbalanced datasets.
5. **Execute One-Layer Stacking:** Combine predictions from Decision Trees, Random Forest, Support Vector Machines, Multilayer Perceptron, and K-Nearest Neighbors into one-layer stacking, exploring different combinations for optimal performance.
6. **Optional Hyperparameter Tuning:** Conduct hyperparameter tuning for the stacked model to optimize parameters and enhance overall performance.

# Dataset
The dataset for this project revolves around the Homesite Quote Conversion Kaggle competition. It is a binary classification challenge focused on predicting the probability of a customer purchasing a quoted 
insurance plan. The competition, which concluded seven years ago, offers a comprehensive set of preprocessed datasets. These datasets have undergone necessary treatments such as the removal of columns with 
abundant missing values and imputation for select missing values. To access and explore the dataset, you can visit [the Homesite Quote Conversion Kaggle competition](https://drive.google.com/drive/folders/10b00DLBT8pxkQIdOi9xch42KwEySoszQ?usp=sharing). The provided preprocessed datasets include 
training and testing sets in CSV format, offering a foundation for implementing advanced machine learning techniques.

# Python Packages Used
1. vecstack
2. pandas
3. numpy
4. plotly.express
5. accuracy_score, classification_report, confusion_matrix, roc_auc_score from sklearn.metrics
6. train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score from sklearn.model_selection
7. RandomForestClassifier, GradientBoostingClassifier from sklearn.ensemble
8. DecisionTreeClassifier from sklearn.tree
9. MLPClassifier from sklearn.neural_network
10. SMOTE from imblearn.over_sampling
11. OneHotEncoder from sklearn.preprocessing

# Result
The project outcomes include refined predictive models that leverage advanced techniques such as SMOTE for handling imbalanced data and ensemble predictions through stacking. These methods enhance accuracy and effectiveness in predicting insurance plan purchases. Comprehensive metrics, including classification reports, confusion matrices, and ROC-AUC scores, demonstrate the improved performance of these models. Validated through cross-validation, the refined models show significant performance improvements in the Kaggle competition. The submission-ready predictions contribute to a deeper understanding of the Homesite Quote Conversion challenge.
