#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:35:54 2024

@author: ali
"""


import os
import re
import pandas as pd
import numpy as np
import scipy
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, make_lsq_spline
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib import cm
from scipy.stats import percentileofscore
import seaborn as sns
from scipy.stats import linregress, t, stats
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from statannotations.Annotator import Annotator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from metric_learn import NCA
from scipy.stats import norm  # Import norm from scipy.stats

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Function to z-score a single matrix
def z_score_matrix(matrix):
    mean = np.mean(matrix)
    std_dev = np.std(matrix)
    z_scored_matrix = (matrix - mean) / std_dev
    return matrix
    #return z_scored_matrix
    
def threshold_matrix(matrix, percentile=30):
    # Flatten the matrix to sort and find the threshold value
    flat_matrix = matrix.flatten()
    threshold_value = np.percentile(flat_matrix, percentile)  # Get the 20th percentile value
    # Apply threshold
    matrix[matrix < threshold_value] = 0
    #print(threshold_value)
    return matrix
    
# Directory containing the CSV files
directory = '/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/connectome/DTI/plain/'

subj_list = []
matrices = []
matrix_size = (84,84)

# Iterate through files in the directory
for filename in os.listdir(directory):
    if re.search("\d_conn_plain.csv$",filename):
        file_path = os.path.join(directory, filename)
        # Read CSV file into a dataframe
        subj_df = pd.read_csv(file_path, header=None)  # Assuming there is no header
        if subj_df.shape == matrix_size:
            subj_list.append(int(os.path.basename(file_path)[4:8]))
            z_scored_matrix = z_score_matrix(subj_df.to_numpy())
            thresholded_matrix = threshold_matrix(z_scored_matrix, 10)
            matrices.append(thresholded_matrix)
df_conn_plain = pd.DataFrame(index=subj_list)
df_conn_plain['matrices']  = matrices
    






df_metadata = pd.read_excel("/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/metadata/alex-badea_2024-06-12.xlsx")
##### drop ones without biomarkers
df_metadata = df_metadata.dropna(subset=['PTAU181'])
len(df_metadata)
### aslo ratio of ab
df_metadata ['ABratio']  =  (df_metadata['AB42'] / df_metadata['AB40'] )
df_metadata ['ABratio']  =  df_metadata ['ABratio'] / np.max(df_metadata ['ABratio'])
df_metadata ['PTAU181']  =  df_metadata ['PTAU181'] / np.max(df_metadata ['PTAU181'])
df_metadata ['NFL']  =  df_metadata ['NFL'] / np.max(df_metadata ['NFL'])




df_metadata['PTID'] =  df_metadata['PTID'].str.slice(start=4 , stop = 8 ).apply(int)


df_metadata.set_index('PTID', inplace=True)

df = df_conn_plain.join(df_metadata, how='inner')

##### learning distance parameter

df['DEMENTED'].fillna(0, inplace=True)

X = np.array( [matrix.flatten() for matrix in df.matrices] )
y = np.array( df.DEMENTED)



nca = NCA(random_state=2)
nca.fit(X, y)

metric_fun = nca.get_metric()

metric_fun( X[1],X[2])

#########
X = np.array(df[['ABratio' , 'PTAU181', 'NFL'] ]  )
y = np.array( df.DEMENTED)



nca2 = NCA(random_state=2)
nca2.fit(X, y)

metric_fun2 = nca2.get_metric()

#metric_fun( X[1],X[2])










# Function to calculate Wasserstein distance between two matrices
def calculate_distance(matrix1, matrix2):
    return wasserstein_distance(matrix1.flatten(), matrix2.flatten())

def matrix_of_distances(matrices_a, matrices_b):
    distances = np.zeros((len(matrices_a), len(matrices_b)))
    for i, matrix_a in enumerate(matrices_a):
        for j, matrix_b in enumerate(matrices_b):
            distances[i, j] = metric_fun(matrix_a.flatten(), matrix_b.flatten())
    return distances



def biom_of_distances(df1, df2):
    distance_matrix = np.zeros((df1.shape[0], df2.shape[0]))
    for i in range(df1.shape[0]):
        for j in range(df2.shape[0]):
            distance_matrix[i, j] = metric_fun2(df1.iloc[i], df2.iloc[j])
    return distance_matrix

# Filter rows where 'risk_for_ad' is 2 or 3
risky = df[df['DEMENTED'].isin([1])]
print(risky)




df_distances = pd.DataFrame(matrix_of_distances(df['matrices'],risky['matrices']),index=df.index, columns=risky.index)




##### bio distance


df_distances_bio = pd.DataFrame(biom_of_distances(df[['ABratio' , 'PTAU181', 'NFL'] ], risky[['ABratio' , 'PTAU181', 'NFL'] ]),index=df.index, columns=risky.index)


df_distances['average_distance'] = df_distances.mean(axis=1)
df = df.join(df_distances['average_distance'])

df_distances_bio['average_distance_bio'] = df_distances_bio.mean(axis=1)
df = df.join(df_distances_bio['average_distance_bio'])


filtered = df 


simmilarity_both= (1/filtered['average_distance']) / np.max(1/filtered['average_distance']) + (1/filtered['average_distance_bio']) / np.max(1/filtered['average_distance_bio'])   #(1/filtered.average_distance)/max(1/filtered.average_distance)
simmilarity_both = simmilarity_both /2
simmilarity_bio=  (1/filtered['average_distance_bio']) / np.max(1/filtered['average_distance_bio'])   #(1/filtered.average_distance)/max(1/filtered.average_distance)
simmilarity_conn = (1/filtered.average_distance)/max(1/filtered.average_distance)



slope, intercept, r_value, p_value, std_err = linregress(filtered.SUBJECT_AGE_SCREEN, simmilarity_both)
predicted = intercept + slope * filtered['SUBJECT_AGE_SCREEN']
filtered['simmilarity_both'] =   simmilarity_both - intercept + slope*filtered.SUBJECT_AGE_SCREEN


slope, intercept, r_value, p_value, std_err = linregress(filtered.SUBJECT_AGE_SCREEN, simmilarity_bio)
predicted = intercept + slope * filtered['SUBJECT_AGE_SCREEN']
filtered['simmilarity_bio'] =  simmilarity_bio - intercept + slope*filtered.SUBJECT_AGE_SCREEN

slope, intercept, r_value, p_value, std_err = linregress(filtered.SUBJECT_AGE_SCREEN, simmilarity_conn)
predicted = intercept + slope * filtered['SUBJECT_AGE_SCREEN']
filtered['simmilarity_conn'] = simmilarity_conn- intercept + slope*filtered.SUBJECT_AGE_SCREEN






# # Define predictors and target
# predictors = ['simmilarity_bio', 'simmilarity_conn', 'simmilarity_both']
# y = filtered['DEMENTED'].values  # Target variable

# # Initialize a plot
# plt.figure()

# # Colors for different ROC curves
# colors = ['blue', 'green', 'red']

# # Train a logistic regression model and plot ROC curve for each predictor
# for predictor, color in zip(predictors, colors):
#     X = filtered[[predictor]].values  # Predictor variable
    
#     # Train the model
#     logreg = LogisticRegression()
#     logreg.fit(X, y)
    
#     # Make predictions and compute probabilities
#     y_pred_prob = logreg.predict_proba(X)[:, 1]
    
#     # Calculate the ROC curve and AUC
#     fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
#     roc_auc = roc_auc_score(y, y_pred_prob)
    
#     # Plot the ROC curve
#     plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve for {predictor} (area = {roc_auc:.2f})')

# # Plot the diagonal line for reference
# plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

# # Configure plot
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curves')
# plt.legend(loc="lower right")
# plt.savefig('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/plots/ROC_demented.png', dpi=300)
# plt.show()




# df['APOE'].fillna(0, inplace=True)
# filtered.loc[(filtered['APOE'] == '2/3') | (filtered['APOE'] == '3/3'), 'APOE'] = 'E33'
# filtered.loc[(filtered['APOE'] == '2/4') | (filtered['APOE'] == '3/4')| (filtered['APOE'] == '4/4'), 'APOE'] = 'E44'



# filteredapoe = filtered[filtered['APOE'] != 0]

# label_encoder = LabelEncoder()
# filteredapoe['APOE_binary'] = label_encoder.fit_transform(filteredapoe['APOE'])


# # Define predictors and target
# predictors = ['simmilarity_bio', 'simmilarity_conn', 'simmilarity_both']
# y = filteredapoe['APOE_binary'].values  # Target variable

# # Initialize a plot
# plt.figure()

# # Colors for different ROC curves
# colors = ['blue', 'green', 'red']

# # Train a logistic regression model and plot ROC curve for each predictor
# for predictor, color in zip(predictors, colors):
#     X = filteredapoe[[predictor]].values  # Predictor variable
    
#     # Train the model
#     logreg = LogisticRegression()
#     logreg.fit(X, y)
    
#     # Make predictions and compute probabilities
#     y_pred_prob = logreg.predict_proba(X)[:, 1]
    
#     # Calculate the ROC curve and AUC
#     fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
#     roc_auc = roc_auc_score(y, y_pred_prob)
    
#     # Plot the ROC curve
#     plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve for {predictor} (area = {roc_auc:.2f})')

# # Plot the diagonal line for reference
# plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

# # Configure plot
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curves')
# plt.legend(loc="lower right")
# plt.savefig('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/plots/ROC_geno.png', dpi=300)
# plt.show()








# label_encoder = LabelEncoder()
# filtered['SUBJECT_SEX_binary'] = label_encoder.fit_transform(filtered['SUBJECT_SEX'])


# # Define predictors and target
# predictors = ['simmilarity_bio', 'simmilarity_conn', 'simmilarity_both']
# y = filtered['SUBJECT_SEX_binary'].values  # Target variable

# # Initialize a plot
# plt.figure()

# # Colors for different ROC curves
# colors = ['blue', 'green', 'red']

# # Train a logistic regression model and plot ROC curve for each predictor
# for predictor, color in zip(predictors, colors):
#     X = filtered[[predictor]].values  # Predictor variable
    
#     # Train the model
#     logreg = LogisticRegression()
#     logreg.fit(X, y)
    
#     # Make predictions and compute probabilities
#     y_pred_prob = logreg.predict_proba(X)[:, 1]
    
#     # Calculate the ROC curve and AUC
#     fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
#     roc_auc = roc_auc_score(y, y_pred_prob)
    
#     # Plot the ROC curve
#     plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve for {predictor} (area = {roc_auc:.2f})')

# # Plot the diagonal line for reference
# plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

# # Configure plot
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curves')
# plt.legend(loc="lower right")
# plt.savefig('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/plots/ROC_sex.png', dpi=300)
# plt.show()




#####################




df['APOE'].fillna(0, inplace=True)
filtered.loc[(filtered['APOE'] == '2/3') | (filtered['APOE'] == '3/3'), 'APOE'] = 'E33'
filtered.loc[(filtered['APOE'] == '2/4') | (filtered['APOE'] == '3/4')| (filtered['APOE'] == '4/4'), 'APOE'] = 'E44'



filteredapoe = filtered[filtered['APOE'] != 0]

label_encoder = LabelEncoder()
filteredapoe['APOE_binary'] = label_encoder.fit_transform(filteredapoe['APOE'])





# Define predictors and target
predictors = ['simmilarity_bio', 'simmilarity_conn', 'simmilarity_both']
y = filteredapoe['APOE_binary'].values  # Target variable

# Initialize a plot
plt.figure()

# Colors for different ROC curves
colors = ['blue', 'green', 'red']

# Store AUCs and their confidence intervals
auc_values = {}
confidence_intervals = {}

# Number of bootstrap samples
n_bootstraps = 1000

# Train a logistic regression model and plot ROC curve for each predictor
for predictor, color in zip(predictors, colors):
    X = filteredapoe[[predictor]].values  # Predictor variable
    
    # Train the model
    logreg = LogisticRegression()
    logreg.fit(X, y)
    
    # Make predictions and compute probabilities
    y_pred_prob = logreg.predict_proba(X)[:, 1]
    
    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = roc_auc_score(y, y_pred_prob)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve for {predictor} (area = {roc_auc:.2f})')
    
    # Store the AUC value
    auc_values[predictor] = roc_auc
    
    # Bootstrap AUC
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(X), len(X))
        if len(np.unique(y[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        
        score = roc_auc_score(y[indices], y_pred_prob[indices])
        bootstrapped_scores.append(score)
    
    # Compute the confidence interval
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    confidence_intervals[predictor] = (ci_lower, ci_upper)
    
    # Z-test for hypothesis testing AUC = 0.5 vs AUC != 0.5
    se_auc = np.std(bootstrapped_scores)
    z = (roc_auc - 0.5) / se_auc
    p_value = 2 * (1 - norm.cdf(np.abs(z)))  # two-tailed test
    
    print(f'Predictor: {predictor}, AUC: {roc_auc:.3f}, CI: [{ci_lower:.3f}, {ci_upper:.3f}], p-value: {p_value:.3f}')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

# Configure plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.savefig('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/plots/ROC_APOE.png', dpi=300)
plt.show()






