#functions
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def age_groups(x):
    '''This function is made for the age column. It creates age bins from the column.'''
    if x in range(20, 30):
        return '2'
    elif x in range(30, 40):
        return '3'
    else:
        return '4'
        
def impute(df, cols):
    '''This function takes the dataframe and list of columns. It imputes the -2 values with 0, the mode, for the respective columns'''
    for i in cols:
        df[i]=np.where(df[i]=='-2','0',df[i])
    return df
    
def default_outlier(df, colnum1, colnum2):
    """
    This function takes the dataframe and range of columns to calculate the 5 std threshold for outliers for specified columns,
    and returns a list of tuples with (column(DEFAULT < 5sd, total < 5sd),(DEFAULT > 5sd, total > 5sd))
    """
    #getting the columns of interest
    cols = df.columns[colnum1:colnum2]
    
    #getting the range limits below and above 5std for each of the columns
    range_limits = []
    for i in cols:
        range_limits.append((df[i].mean() - (df[i].std()*5),
                             df[i].mean() + (df[i].std()*5)))
    #making a list of tuples for number of defaulters past the upper and lower bounds for each column
    default_over_thresh = {}
    for i, n in enumerate(cols):
        default_over_thresh[n] = [(df[(df[n] < range_limits[i][0]) & (df['DEFAULT'] == '1')].shape[0], df[df[n] < range_limits[i][0]].shape[0])
             ,(df[(df[n] > range_limits[i][1]) & (df['DEFAULT'] == '1')].shape[0], df[df[n] > range_limits[i][1]].shape[0])]
        
    #returning the percent of defaults over the threshold
    return default_over_thresh

def knn_creator(k, X_train_scale, y_train):
    """This function will be used to create a K Nearest Neighbors model with a specified number of k"""
    #make an instance of KNN, the default is k=5
    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train_scale, y_train)
    
    return knn