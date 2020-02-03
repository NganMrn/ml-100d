##### Data PreProcessing #####
## Step 1: Importing the libraries
import numpy as np
import pandas as pd

## Step 2: Importing dataset
dataset = pd.read_csv('datasets/Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
print("import data ------------------------------------------------")
print(X)
print("-------------------------------------------------------------")
print(Y)
print("app running ------------------------------------------------")

## Step 3: Handling the missing data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

## Step 4: Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

### Creating a dummy variable
#onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = OneHotEncoder(categories='auto')
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

## Step 5: Splitting the datasets into training sets and Test sets 
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

## Step 6: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_train)
print("_______________________________________________")
print(X_test)

### Done :smile:

#############- NOTE -######################
### As you can see the init does not get the variable categorical_features
# You have an categories flag:
# categories‘auto’ or a list of array-like, default=’auto’ Categories (unique values) per feature:
# ‘auto’ : Determine categories automatically from the training data.
# list : categories[i] holds the categories expected in the ith column.
# The passed categories should not mix strings and numeric values within a single feature, and should be sorted in case of numeric values.
# The used categories can be found in the categories_ attribute.
# Attributes: categories_list of arrays The categories of each feature determined during fitting (in order of the features in X and corresponding with the output of transform). This includes the category specified in drop (if any).
### sklearn.cross_validation now it's in the model_selection
# from sklearn.model_selection import train_test_split