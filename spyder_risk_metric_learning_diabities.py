#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:56:24 2024

@author: ali
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:11:42 2024

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
    






df_metadata = pd.read_excel("/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/ADRC/metadata/alex-badea_2024-06-14.xlsx")
##### drop ones without biomarkers
df_metadata = df_metadata.dropna(subset=['PTAU181'])
len(df_metadata)
### aslo ratio of ab
df_metadata ['ABratio']  =  (df_metadata['AB42'] / df_metadata['AB40'] )
df_metadata ['ABratio']  =  df_metadata ['ABratio'] / np.max(df_metadata ['ABratio'])

df_metadata ['PTAU181']  =  df_metadata ['PTAU181'] / np.max(df_metadata ['PTAU181'])

df_metadata ['NFL']  =  df_metadata ['NFL'] / np.max(df_metadata ['NFL'])

#df_metadata ['GFAP']  =  df_metadata ['GFAP'] / np.max(df_metadata ['GFAP'])



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

#df_distances_bio = pd.DataFrame(biom_of_distances(df[['ABratio' ] ], risky[['ABratio' ] ]),index=df.index, columns=risky.index)



# df_distances
# df_distances.to_csv('df_distances.csv')


df_distances['average_distance'] = df_distances.mean(axis=1)
df = df.join(df_distances['average_distance'])

df_distances_bio['average_distance_bio'] = df_distances_bio.mean(axis=1)
df = df.join(df_distances_bio['average_distance_bio'])



#df['genotype_grouped'] = df['genotype'].map({'APOE23':'APOE23/33', 'APOE33':'APOE23/33', 'APOE34':'APOE34/44','APOE44':'APOE34/44'})

#df

filtered = df #df[~df['DEMENTED'].isin([1])]


fig = px.scatter(filtered, x='SUBJECT_AGE_SCREEN', y=1/filtered.average_distance, trendline='ols', title='Age vs 1/distance')
fig.show()




# # plot with the spline
# filtered = filtered.sort_values('SUBJECT_AGE_SCREEN')


# spl = UnivariateSpline(filtered.SUBJECT_AGE_SCREEN, 1/filtered.average_distance,k=3)
# plt.plot(filtered.SUBJECT_AGE_SCREEN, spl(filtered.SUBJECT_AGE_SCREEN), 'g', lw=3)
# #colors = {"APOE23/33":"blue","APOE34/44":"red"}
# #colors = {"APOE23/33":"blue","APOE34/44":"red"}
# #plt.scatter(filtered.age, 1/filtered.average_distance, c=filtered['genotype_grouped'].map(colors))
# plt.ylabel('1/Wasserstein Scaled')
# plt.xlabel('Age')
# plt.title('Linear Fit')
# plt.legend()
# plt.show()

simmilarity= (1/filtered['average_distance']) / np.max(1/filtered['average_distance']) + (1/filtered['average_distance_bio']) / np.max(1/filtered['average_distance_bio'])   #(1/filtered.average_distance)/max(1/filtered.average_distance)
simmilarity = simmilarity /2
# simmilarity=  (1/filtered['average_distance_bio']) / np.max(1/filtered['average_distance_bio'])   #(1/filtered.average_distance)/max(1/filtered.average_distance)
# simmilarity = (1/filtered.average_distance)/max(1/filtered.average_distance)


df['similarity'] = simmilarity

filtered = df #df[~df['DEMENTED'].isin([1])]


percentile_similarity   = [percentileofscore(simmilarity, i) for i in simmilarity]
filtered['percentile_similarity']  = percentile_similarity
#filtered.to_csv('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/code/percentile_similarity.csv')


plt.figure(figsize=(10, 6))
#plt.plot(filtered.SUBJECT_AGE_SCREEN, percentile_similarity, 'g', lw=3)
plt.scatter(filtered.SUBJECT_AGE_SCREEN, percentile_similarity)
#plt.scatter(filtered.age, simmilarity, c=filtered['genotype_grouped'].map(colors))
plt.ylabel('Percentile Simmilarity')
plt.xlabel('Age')
plt.title('Linear Fit')
plt.legend()
#plt.savefig('/Users/ali/Desktop/May24/Alex_Wstein_Risk_model_spline/plots/distance.png', dpi=300)
plt.show()






filtered['DEMENTED'].fillna(0, inplace=True)
filtered['APOE'].fillna(0, inplace=True)
set(filtered['APOE'])

 

filtered.columns








slope, intercept, r_value, p_value, std_err = linregress(filtered.SUBJECT_AGE_SCREEN, simmilarity)
predicted = intercept + slope * filtered['SUBJECT_AGE_SCREEN']





###residuals

filtered['Residuals']  = simmilarity - intercept + slope*filtered.SUBJECT_AGE_SCREEN










filtered.loc[(filtered['APOE'] == '2/3') | (filtered['APOE'] == '3/3'), 'APOE'] = 'E33'
filtered.loc[(filtered['APOE'] == '2/4') | (filtered['APOE'] == '3/4')| (filtered['APOE'] == '4/4'), 'APOE'] = 'E44'

# Define the mapping dictionary
diabetes_mapping = {
    0: 'Absent',
    1: 'Recent/Active',
    2: 'Remote/Inactive', #Remote/Inactive
    9: 'Unknown'
}

# Apply the mapping to the DIABETES column
filtered['DIABETES_DESCR'] = filtered['DIABETES'].map(diabetes_mapping)



# Create a violin plot
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x='DIABETES_DESCR', y='similarity', data=filtered)
plt.title('Violin Plot of Similarity Score by DIABETES Status' )
plt.xlabel('DIABETES')
plt.ylabel('Similarity Score (Connectome + biom )')
# Define the pairs for comparison and the test to be used
#pairs = [ ("Absent" , "Recent/Active") ]
pairs = [ ("Absent" , "Recent/Active") , ("Absent" , "Remote/Inactive"), ("Recent/Active","Remote/Inactive") ]

annotator = Annotator(ax, pairs, data=filtered, x='DIABETES_DESCR', y='similarity')
# Apply the t-test
annotator.configure(test='t-test_ind', text_format='star', loc='inside')
annotator.apply_and_annotate()
plt.savefig('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/plots/conn_bio_diabities.png', dpi=300)
plt.show()
# Create a violin plot





# Create a violin plot
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x='DIABETES_DESCR', y='Residuals', data=filtered)
plt.title('Violin Plot of Similarity Score Residuals by DIABETES Status' )
plt.xlabel('DIABETES')
plt.ylabel('Similarity Score (Connectome + biom )')
# Define the pairs for comparison and the test to be used
 pairs = [ ("Absent" , "Recent/Active") , ("Absent" , "Remote/Inactive"), ("Recent/Active","Remote/Inactive") ]
#pairs = [ ("Absent" , "Recent/Active") ]
annotator = Annotator(ax, pairs, data=filtered, x='DIABETES_DESCR', y='Residuals')
# Apply the t-test
annotator.configure(test='t-test_ind', text_format='star', loc='inside')
annotator.apply_and_annotate()
plt.savefig('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/plots/conn_bio_diabities_resid.png', dpi=300)
plt.show()
# Create a violin plot










slope, intercept, r_value, p_value, std_err = linregress(filtered.SUBJECT_AGE_SCREEN, simmilarity)
predicted = intercept + slope * filtered['SUBJECT_AGE_SCREEN']




# n = len(filtered['SUBJECT_AGE_SCREEN'])
# alpha = 0.05
# t_value = t.ppf(1 - alpha / 2, df=n-2)
# mean_x = np.mean(filtered['SUBJECT_AGE_SCREEN'])
# sum_squares_x = np.sum((filtered['SUBJECT_AGE_SCREEN'] - mean_x) ** 2)
# pred_se = std_err * np.sqrt(1/n + (filtered['SUBJECT_AGE_SCREEN'] - mean_x)**2 / sum_squares_x)
# ci_upper = predicted + t_value * pred_se
# ci_lower = predicted - t_value * pred_se

plt.figure(figsize=(10, 6))
sns.regplot(x=filtered.SUBJECT_AGE_SCREEN, y=simmilarity, data=filtered, color='purple')

#spl = UnivariateSpline(filtered.SUBJECT_AGE_SCREEN, simmilarity,k=3)
#plt.plot(filtered.SUBJECT_AGE_SCREEN, spl(filtered.SUBJECT_AGE_SCREEN), 'g', lw=3)
plt.plot(filtered.SUBJECT_AGE_SCREEN, intercept + slope*filtered.SUBJECT_AGE_SCREEN, 'g', lw=3)
#plt.fill_between(filtered['SUBJECT_AGE_SCREEN'], ci_lower, ci_upper, color='b', alpha=1, label='95% Confidence Interval')
# plt.scatter(filtered.SUBJECT_AGE_SCREEN, simmilarity)
colors = {"Recent/Active":"red", "Absent":"blue", "Remote/Inactive" :"green" }
plt.scatter(filtered.SUBJECT_AGE_SCREEN, simmilarity, c=filtered['DIABETES_DESCR'].map(colors))
# filtered['APOE'] = filtered['APOE'].astype(str)
# colors = {"2/4":"red", "3/4":"red", "4/4":"red" ,"2/3":"blue","3/3":"blue" , "0":"gray"}
# plt.scatter(filtered.SUBJECT_AGE_SCREEN, simmilarity, c=filtered['APOE'].map(colors))

#filtered['SUBJECT_SEX'] = filtered['SUBJECT_SEX'].astype(str)
#colors = {"1":"red", "2":"blue"}
#plt.scatter(filtered.SUBJECT_AGE_SCREEN, simmilarity, c=filtered['SUBJECT_SEX'].map(colors))

plt.ylabel('Simmilarity (connectome)')
plt.xlabel('Age')
#plt.title('Linear Fit after threshold')
plt.text(30, 1, 'Intercept='+str(intercept)[0:6] + ' Slope='+str(slope)[0:6] + ' r='+str(r_value)[0:6]  + ' p='+str("{:.3e}".format(p_value))[0:15]  ,fontsize=12)
plt.legend()
# plt.savefig('/Users/ali/Desktop/Jun24/Alex_Wstein_Risk_model_spline_ADRC/plots/conn.png', dpi=300)
plt.show()


