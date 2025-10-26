import numpy as np

#Function to detect outliers through z score
def detect_outliers(df, col, thresh=3):
    mean = np.mean(df[col]) 
    std = np.std(df[col])
    outlier = [] 
    for i in df[col]: 
         z = abs(i-mean)/std #abs needed?
         if z > thresh: 
            outlier.append(i) 
    return outlier    

#Function to remove the outliers 
def remove_outliers(df, col, thresh):
    df = df.drop(df.loc[df[col] > thresh].index) 
    return df  