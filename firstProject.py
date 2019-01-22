# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# Load data from .csv
data = pd.read_csv("D:/Data Science/japan-population-data/Japan_population_data.csv", sep=",")

#data cleaning(i.e, replacing the values = 'NA' with 0)
data.fillna(0,inplace=True)

#filtering data for one column
data_filtered = data[data.prefecture == 'Aichi-ken']
data_filtered_cols = data_filtered[['year','population','estimated_area']]

#assigning X and y values and train and test 
X = data_filtered_cols[['year','estimated_area']]
y = data_filtered_cols['population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#linear regression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)