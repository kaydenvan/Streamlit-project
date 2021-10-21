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
**Classification** is a process of categorizing a given set of data into classes. It is a fundamental with a wide usage supervised machine learning, allowing Data Analyst/Scientist to have a quick understanding to train the model. Sklearn, XGBoost, CatBoost and LightGBM are the core libraries used for this module and an automated classification process has been created.<br><br>
**File Input**: .xls/.xlsx/.csv file format is required while header should be included in the first row of the file.<br><br>
This module will provide you <br>
1. Model setup<br>
1.1. Choose training vs testing size ratio <br>
1.2. Choose target variable<br>
1.3. Choose features for model training<br>
2. Data cleansing<br>
2.1. Fill up null values with most frequent or median values depends on data types<br>
2.2. Standardize numeric columns<br>
2.3. One-hot encoding for categorical columns<br>
3. Model configuration<br>
3.1. Select model(s) to develop. Logistic Regression, Random Forest, LightGBM, XGBoost and CatBoost are options<br>
4. Model result<br>
4.1. Training set accuracy<br>
4.2. Testing set accuracy<br>
4.3. Testing set precision<br>
4.4. Testing set recall<br>
4.5. Performance matrix of the model<br>
4.6. Confusion matrix of the model<br>
4.7. Preview of result dataset containing the actual and predicted result<br>


### Stock Time Series Prediction (Auto)
**Trending** is a very common usecase in real life. For financial institutes, revenue and sales forecast are common questions by management for better decision making. In this module, US stock is used and provide a real=time training and prediction as time series machine learning showcase. Yahoo Finance and FBProphet are core libraries used for this module.<br><br>
**Data Input**: A valid US stock number (E.g. AAPL, AMZN, IBM and etc.) is required. You could choose different time period of historical stock record.<br><br>
This module will provide you<br>
1. Current stock performance<br>
2. Model result<br>
2.1. Up to 60 days prediction<br>
2.2. Upper and lower bound of predicted value<br>
2.3. Breakdown of seasonal model components from daily to monthly<br>


### Customer Segmentation (Auto)
**Acquisition** and **Retention** are another common usecases across different industries, particularly for retail industry. Customization is one of the strategies on customer acquisition. By clustering existing customers, company could understand their existing customer group and analyze their target group. Sklearn is the core library for this module and an automated clustering process has been created.<br><br>
**File Input**: .xls/.xlsx/.csv file format is required while header should be included in the first row of the file.<br><br>
This module will provide you <br>
1. Model configuration<br>
1.1. Principal components analysis(PCA)<br>
1.2. Elbow Method<br>
1.3. Silhouette Coefficients<br>
1.4. Data Visualization<br>
2. Data cleansing<br>
2.1. Fill up null values with most frequent or median values depends on data types<br>
2.2. Standardize numeric columns<br>
2.3. Ordinal object columns<br>
3. Model result<br>
3.1. Visualize Elbow graph<br>
3.2. Visualize Silhouette Coefficients<br>
3.3. 2D visualization of user selected components<br>


### XGBoost classification model with detail analysis (Auto)
### XGBoost Regression models (For Kaggle)
