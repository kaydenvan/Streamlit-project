# Streamlit Project
Streamlit is an open-source Python library which enable users to build a custom web apps, mainly for machine learning and data science, quickly and beautifully. As it gives a brillant chance to demonstrate and deploy solutions freely and quickly, I created this repository to showcase usual machine learning use cases. This project is published to [Webpage here](https://share.streamlit.io/kaydenvan/streamlit_development/__main__.py).

#### 6 modules have been created.
* Exploratory Data Analysis
* Categorical Prediction (Auto)
* Stock Time Series Prediction (Auto)
* Customer Segmentation (Auto)
* XGBoost classification model with detail analysis (Auto)
* XGBoost Regression models (For Kaggle)

### Exploratory Data Analysis(EDA)
Before any data analysis starts, exploratory data analysis is the key to drive success. 
It is very important to understand your data to provide great and accurate insights.<br><br>
Saying that, **Data Visualization** is the core part of this module. 
Matplotlib and Seaborn are mainly used and an automated EDA process has been created.<br><br>
**File Input**: .xls/.xlsx/.csv file format is required while header should be included in the first row of the file.<br><br>
This module will provide you <br>
1. Basic statistical information (E.g. dataframe size, mean, standard deviation, quartile and etc.)for each numerical columns<br>
2. Any null column found in the dataset. If yes, the total number of null value in each column will be shown<br>
3. Column type for each column<br>
4. Data Visualization<br>
4.1. Categorical values by Histogram<br>
4.2. Numeric values by using Bar Chart, Kernel Density Estimation and Box Plot<br>
5. Data Correlation<br>
5.1. High correlation parameters will be shown<br>
5.2. Correlation Matrix<br>

### Categorical Prediction
After having a brief understanding of the dataset, it is good to put it to some common machine learning algorithms to see how it perform. 
Sklearn, XGBoost, CatBoost and LightGBM are the core libraries used in this module<br><br>
**File Input**: .xls/.xlsx/.csv file format is required while header should be included in the first row of the file.<br><br>
This module will provide you <br>
1. Model configuration
1.1. Choose training vs testing size ratio 
1.2. Choose target variable
1.3. Choose features for model training
2. Data cleansing
2.1. Fill up null values with most frequent or median values depends on data types
2.2. Standardize numeric columns
2.3. One-hot encoding for categorical columns
3. Model configuration
3.1. Select model(s) to develop. Logistic Regression, Random Forest, LightGBM, XGBoost and CatBoost are options
4. Model result
4.1. Training set accuracy
4.2. Testing set accuracy
4.3. Testing set precision
4.4. Testing set recall
4.5. Performance matrix of the model
4.6. Confusion matrix of the model
4.7. Preview of result dataset containing the actual and predicted result


### Stock Time Series Prediction (Auto)
### Customer Segmentation (Auto)
### XGBoost classification model with detail analysis (Auto)
### XGBoost Regression models (For Kaggle)
