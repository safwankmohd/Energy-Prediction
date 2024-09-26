# Energy Prediction

## Overview

Energy consumption patterns are vital for resource planning, optimizing utility management, and developing sustainable energy policies. Communities rely on accurate data about different energy sources, such as electricity and natural gas, to make informed decisions. However, predicting the type of energy being consumed based on regional, seasonal, and utility-based factors is challenging due to the diverse and complex interactions between these variables. By developing an effective classification model to predict the type of energy consumption, we can enhance energy planning efforts, improve infrastructure management, and promote efficient resource utilization across various communities.

## Dataset

The dataset includes the following features:
- year
- data_class
- month
- value
- com_name
- com_type
- data_field
- com_county
- geometry_id
- full_fips
- unit
- uer_id
- data_stream
- utility_display_name
- number_of_accounts
-  Georeference

## Methodology

The following steps were taken to build the heart attack risk prediction model:

1. **Data Preprocessing**:
   - Handling missing values.
   - Feature scaling.
   - Encoding categorical variables.

2. **Feature Engineering**:
   - Selecting the top 10 important features using `RandomForestClassifier`.

3. **Model Building**:
   - Implementing several classification algorithms including LogisticRegression, svm, DecisionTree, RandomForest and aive_bayes
   - Hyperparameter tuning to optimize model performance.

4. **Model Evaluation**:
   - Evaluating models using metrics like accuracy, precision, recall, and F1-score.
   - Using confusion matrix to assess model performance.


## Model Performance

The RandomForest model was optimized with the following parameters:

- `n_estimators=100`
- `max_depth=30`  
- `min_samples_split=2`
- `min_samples_leaf=1`  
- `bootstrap'= False`  
- `random_state=42`


## How to Use

To use the model to predict heart attack risk for new data, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/heart-attack-risk-prediction.git
    cd heart-attack-risk-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Load the trained model and scaler:
    ```python
    import pickle
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    with open('Energy_prediction.pkl', 'rb') as file:
        rf1 = pickle.load(file)

    with open('Energy_prediction.pkl', 'rb') as file:
        scaler = pickle.load(file)
    ```

4. Create a DataFrame with the new data and make predictions:
    ```python
    numerical_features = [ 'value', 'number_of_accounts']  # Adjust these based on your dataset

    unseen_data = pd.DataFrame({
    'value': [5000.0], 
    'com_name': ['Sodus'],  
    'com_type': ['Village'],  
    'data_field': ['all_other_(o)'], 
    'com_county': ['Wayne'],
    'unit': ['MWh'], 
    'number_of_accounts': [50], 
    'Georeference': ['POINT (-77.061462 43.236085)']
    })

    unseen_data_encoded = pd.get_dummies(unseen_data, columns=[ 'com_type', 'unit', ], drop_first=True)
    
    for col in ['com_name', 'Georeference', 'com_county', 'data_field']:
    freq_encoding = X[col].value_counts(normalize=True)  # Calculate the frequency of each category
    unseen_data_encoded[col] = X[col].map(freq_encoding)  # Map the original values to their frequency

    unseen_data_encoded = unseen_data_encoded.reindex(columns=X_train.columns, fill_value=0)
    
    scaler = StandardScaler()
    unseen_data_encoded[numerical_features] = scaler.fit_transform(unseen_data_encoded[numerical_features])
    predicted_class = loaded_model.predict(unseen_data_encoded)
    print("Predicted class:", predicted_class[0])

    ```

## Conclusion

This project demonstrated a strong balance between training and test performance, showing robustness without significant overfitting. The model effectively captured the complex patterns in the dataset and maintained high accuracy across various classes. Despite the presence of missing data and potential class imbalance, the model performed well overall, making it a reliable choice for classification tasks in this dataset. Further improvements could be made by addressing limitations such as class imbalance and enhancing feature engineering.


