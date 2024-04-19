#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv(r'C:\Users\Lenovo\Desktop\prosperLoanData.csv')


# In[3]:


df


# In[4]:


# Data Cleaning


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.info


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


print(df.dtypes)


# In[13]:


df.dropna(inplace=True)
print(df)


# In[14]:


# Handel Missing Values


# In[15]:


df.dropna(inplace=True)
df.fillna(df.mean(), inplace=True)
df


# In[16]:


# Data cleaning entails identifying and rectifying errors, inconsistencies, and missing values in a dataset through steps such as handling missing values by either removing them or filling them using appropriate techniques,eliminating duplicate entries, managing outliers by either removing, capping, or transforming them,standardizing or normalizing data to bring features to a similar scale, encoding categorical variablesinto numerical format, performing feature engineering to create new features or transform existing ones,rectifying typos and inconsistencies, ensuring correct data types, validating data against known constraints,and documenting the entire process to ensure reproducibility and transparency.






# In[17]:


# Data Encoding


# In[18]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[19]:


# This code snippet creates a LabelEncoder object and utilizes it to convert categorical columns within a DataFrame into numerical representations by transforming their values


# In[20]:


import pandas as pd

# Assuming 'df' is your DataFrame containing the dataset
# Let's say you want to encode the 'BorrowerState' column

# Option 1: Label Encoding
# This assigns a unique numerical label to each category
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['BorrowerState_encoded'] = label_encoder.fit_transform(df['BorrowerState'])

# Option 2: One-Hot Encoding
# This creates binary columns for each category
# Note: One-Hot Encoding can be memory-intensive for large datasets with many unique categories
one_hot_encoded = pd.get_dummies(df['BorrowerState'], prefix='BorrowerState')

# Concatenate the one-hot encoded columns with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

# Now 'df' contains the original 'BorrowerState' column encoded using both Label Encoding and One-Hot Encoding


# In[21]:


# This code illustrates two techniques for encoding categorical data found in the 'BorrowerState' column of a DataFrame: Label Encoding, which assigns unique numerical labels to each category, and One-Hot Encoding, which generates binary columns for each category, leading to a sparse matrix representation.


# In[22]:


from sklearn.preprocessing import LabelEncoder

# Assuming your dataset is stored in a DataFrame called 'df'

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column contains categorical data
        df[column] = label_encoder.fit_transform(df[column])

# Now, your categorical columns are encoded with numerical values


# In[23]:


# This code utilizes LabelEncoder from scikit-learn to encode categorical columns in a DataFrame into numerical values, iterating through each column and transforming it if it contains categorical data.


# In[24]:


from sklearn.preprocessing import LabelEncoder

# Assuming your dataset is stored in a DataFrame called 'df'

# Define a list of categorical column names
categorical_columns = ['CreditGrade', 'LoanStatus', 'BorrowerState', 'Occupation', 'EmploymentStatus']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Now, your categorical columns are encoded with numerical labels


# In[25]:


# This code snippet uses LabelEncoder from scikit-learn to encode categorical columns in a DataFrame into numerical labels. It iterates through a predefined list of categorical column names and applies label encoding to each column, replacing categorical values with numerical labels.


# In[26]:


# Label Encoding


# In[27]:


from sklearn.preprocessing import LabelEncoder

# Define the list of categorical column values
LoanStatus = ['Completed', 'Current', 'Completed', 'Current', 'Current', 'Current', 'Current', 'Current']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'LoanStatus' column
encoded_labels = label_encoder.fit_transform(LoanStatus)

# Print the encoded labels
print(encoded_labels)


# In[28]:


# This code snippet demonstrates how to use LabelEncoder from scikit-learn to encode categorical values in the 'LoanStatus' list into numerical labels. After initializing the LabelEncoder object, it fits and transforms the 'LoanStatus' column, replacing the categorical values with their corresponding numerical labels, and then prints the encoded labels.


# In[29]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(your_categorical_data)


# In[ ]:


# One-Hot Encoding: If there is no ordinal relationship between categories, one-hot encoding is typically used. It creates a binary column for each category, where 1 indicates the presence of the category and 0 its absence.


# In[ ]:


import pandas as pd

# Define your categorical data
LoanStatus = ['Completed', 'Current', 'Completed', 'Current', 'Current', 'Current', 'Current', 'Current']

# Convert the categorical data to a pandas DataFrame
your_categorical_data = pd.DataFrame({'LoanStatus': LoanStatus})

# One-hot encode the categorical data
encoded_data = pd.get_dummies(your_categorical_data)

# Display the encoded data
print(encoded_data)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Assuming your_categorical_data is a pandas DataFrame column
encoded_data = pd.get_dummies(your_categorical_data)


# In[ ]:


# Binary Encoding 


# In[ ]:


pip install category_encoders


# In[ ]:


import category_encoders as ce
import pandas as pd

# Define your categorical data
your_data = pd.DataFrame({'your_categorical_column': ['Completed', 'Current', 'Completed', 'Current', 'Current', 'Current', 'Current', 'Current']})

# Initialize the BinaryEncoder
binary_encoder = ce.BinaryEncoder(cols=['your_categorical_column'])

# Encode the categorical data
encoded_data = binary_encoder.fit_transform(your_data)

# Display the encoded data
print(encoded_data)


# In[ ]:


# This code snippet utilizes the BinaryEncoder from the category_encoders library to perform binary encoding on categorical data. It first creates a DataFrame called your_data with a single column 'your_categorical_column' containing categorical values. Then, it initializes a BinaryEncoder object specifying the column to encode. Next, it applies the encoding transformation to the data using the fit_transform() method, resulting in a new DataFrame called encoded_data. Finally, it prints the encoded data. Binary encoding converts each category into binary digits and represents them in separate columns.


# In[37]:


print(df.columns)


# In[44]:


import pandas as pd
import category_encoders as ce

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\Lenovo\Desktop\prosperLoanData.csv')

# Specify the column name to encode
categorical_column_name = 'LoanStatus'

# Check if the specified column exists in the DataFrame
if categorical_column_name not in df.columns:
    raise ValueError(f"The column '{categorical_column_name}' does not exist in the DataFrame.")

# Initialize the BinaryEncoder
binary_encoder = ce.BinaryEncoder(cols=[categorical_column_name])

# Encode the data
encoded_data = binary_encoder.fit_transform(df)


# In[ ]:


# Data Labelling


# In[45]:


def transform(feature):
    le=LabelEncoder()
    df[feature]=le.fit_transform(df[feature])
    print(le.classes_)


# In[46]:


cat_df=df.select_dtypes(include='object')
cat_df.columns


# In[48]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset into a DataFrame
# Assuming your dataset is stored in a CSV file named 'your_dataset.csv'
df = pd.read_csv(r'C:\Users\Lenovo\Desktop\prosperLoanData.csv')

# Assuming 'LoanStatus' is the categorical column you want to encode
label_encoder = LabelEncoder()
df['LoanStatus_encoded'] = label_encoder.fit_transform(df['LoanStatus'])

# Displaying the first few rows to verify the encoding
print(df[['LoanStatus', 'LoanStatus_encoded']].head())


# In[ ]:


# DataFrame, then uses scikit-learn's LabelEncoder to encode the 'LoanStatus' categorical column into numerical labels, adding a new column 'LoanStatus_encoded' to the DataFrame to store the encoded values, and finally, it prints the first few rows of the original 'LoanStatus' column alongside the corresponding encoded values for verification.


# In[49]:


import pandas as pd

# Assuming your DataFrame is named cat_df
for col in cat_df.columns:
    # Display unique values in each column
    unique_values = cat_df[col].unique()
    print(f"Column '{col}' has {len(unique_values)} unique values:")
    print(unique_values)
    print("\n")


# In[50]:


for col in cat_df.columns:
    missing_values = cat_df[col].isnull().sum()
    print(f"Column '{col}' has {missing_values} missing values.")


# In[51]:


for col in cat_df.columns:
    unique_values_count = cat_df[col].nunique()
    print(f"Column '{col}' has {unique_values_count} unique values.")


# In[52]:


for col in cat_df.columns:
    data_type = cat_df[col].dtype
    print(f"Data type of column '{col}' is {data_type}.")


# In[53]:


for col in cat_df.columns:
    value_counts = cat_df[col].value_counts()
    print(f"Value counts for column '{col}':\n{value_counts}\n")


# In[54]:


for col in cat_df.select_dtypes(include='number').columns:
    summary_stats = cat_df[col].describe()
    print(f"Summary statistics for column '{col}':\n{summary_stats}\n")


# In[55]:


from sklearn.preprocessing import LabelEncoder

# List of categorical columns to encode
cat_columns = ['CreditGrade', 'LoanStatus', 'BorrowerState', 'Occupation', 'EmploymentStatus', 'IncomeRange']

# Create a label encoder object
label_encoder = LabelEncoder()

# Iterate through each categorical column and encode it
for col in cat_columns:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

# Displaying the first few rows to verify the encoding
print(df[cat_columns + [col + '_encoded' for col in cat_columns]].head())


# In[ ]:


# This code iterates through each column in the DataFrame cat_df, prints the column name along with the count of unique values it contains, displays the unique values themselves, and adds a newline for clarity before moving to the next column


# In[56]:


from sklearn.preprocessing import LabelEncoder

# Assuming your DataFrame is named df
# Assuming cat_df contains only the categorical columns from df
cat_columns = cat_df.columns

label_encoders = {}
for col in cat_columns:
    # Handling missing values by replacing them with a placeholder
    df[col].fillna('missing', inplace=True)
    
    label_encoders[col] = LabelEncoder()
    df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

# Displaying the first few rows to verify the encoding
print(df[[col + '_encoded' for col in cat_columns]].head())


# In[ ]:


# This code snippet encodes categorical columns in the DataFrame df using scikit-learn's LabelEncoder. It iterates through each categorical column in cat_df, handling missing values by replacing them with a placeholder, and then creates a new column suffixed with '_encoded' containing the encoded values. Finally, it prints the first few rows of the encoded columns for verification.


# In[57]:


from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame containing categorical columns
# Selecting only the categorical columns
cat_df = df.select_dtypes(include=['object'])

# Creating a LabelEncoder object
label_encoder = LabelEncoder()

# Label encoding each categorical column
for col in cat_df.columns:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

# Dropping the original categorical columns
df.drop(cat_df.columns, axis=1, inplace=True)

# Displaying the first few rows to verify the encoding
print(df.head())


# In[ ]:


# This code snippet encodes categorical columns in the DataFrame df using scikit-learn's LabelEncoder. It first selects only the categorical columns into a DataFrame called cat_df. Then, it iterates through each categorical column, applies label encoding using LabelEncoder, and adds a new column suffixed with '_encoded' containing the encoded values. After encoding, it drops the original categorical columns from df. Finally, it prints the first few rows of the DataFrame to verify the encoding.


# In[58]:


from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame containing categorical columns
# Selecting only the categorical columns
cat_df = df.select_dtypes(include=['object'])

# Creating a LabelEncoder object
label_encoder = LabelEncoder()

# Label encoding each categorical column
for col in cat_df.columns:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

# Dropping the original categorical columns
df.drop(cat_df.columns, axis=1, inplace=True)

# Displaying the first few rows to verify the encoding
print(df.head())


# In[ ]:


# This code snippet uses scikit-learn's LabelEncoder to transform categorical columns in a DataFrame into numerical representations, dropping the original categorical columns and adding their encoded versions for machine learning purposes


# In[ ]:


# Loan Status


# In[59]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample DataFrame with LoanStatus column
data = {
    'LoanStatus': ['Cancelled', 'Chargedoff', 'Completed', 'Current', 'Defaulted', 'FinalPaymentInProgress', 'PastDue']
}

df = pd.DataFrame(data)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform LoanStatus column
df['LoanStatus_encoded'] = label_encoder.fit_transform(df['LoanStatus'])

# Display encoded DataFrame
print(df)

