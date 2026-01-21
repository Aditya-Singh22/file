#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[3]:


df = pd.read_csv("mumbai_housing_data.csv")
print("First 5 rows of raw data: ")
print(df.head())


# In[4]:


print("\nBasic Info: ")
print(df.info())


# In[5]:


print("\n Summary Statistics: ")
print(df.describe())


# In[6]:


print("\nMissing Value Per Column: ")
print(df.isnull().sum())


# In[7]:


if "Age_yrs" in df.columns:
    df["Age_yrs"] = df["Age_yrs"].fillna(df["Age_yrs"].median())


# In[8]:


num_cols = ["Area_sqft","BHK","Floor","Total_Floors","Dist_Rail_km","Sea_View","Parking","Price_INR"]
for col in num_cols:
    if col in df.columns:
        df[col]= df[col].fillna(df[col].median())


# In[9]:


lower, upper= df["Price_INR"].quantile([0.01,0.99])
df["Price_INR"]= df["Price_INR"].clip(lower,upper)


# In[10]:


plt.figure(figsize=(7,4))
plt.hist(df["Price_INR"]/1e7, bins=30, edgecolor="black")
plt.xlabel("Price (Crores)")
plt.ylabel("Count")
plt.title("Distribution of Apartment Price in Mumbai")
plt.tight_layout()
plt.show()


# In[11]:


plt.figure(figsize=(7, 4))
plt.hist(df["Price_INR"] / 1e7, bins=30, edgecolor="black")
plt.xlabel("Price (Crores)")
plt.ylabel("Count")
plt.title("Distribution of Apartment Prices (Mumbai)")
plt.tight_layout()
plt.show()


# In[12]:


plt.figure(figsize=(7, 4))
plt.hist(df["Area_sqft"], bins=30, edgecolor="black")
plt.xlabel("Area (sq ft)")
plt.ylabel("Count")
plt.title("Distribution of Apartment Area")
plt.tight_layout()
plt.show()


# In[13]:


if "Locality" in df.columns:
    plt.figure(figsize=(8, 5))
    localities = df["Locality"].unique().tolist()
    prices_by_loc = [df[df["Locality"] == loc]["Price_INR"] / 1e7 
                     
for loc in localities]
plt.boxplot(prices_by_loc, labels=localities)
plt.ylabel("Price (Crores)")
plt.title("Price Distribution by Locality")
plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(7, 5))
plt.scatter(df["Area_sqft"], df["Price_INR"] / 1e7, alpha=0.5)
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Crores)")
plt.title("Area vs Price")
plt.tight_layout()
plt.show()


# In[20]:


numeric_cols = [
    "Area_sqft",
    "BHK",
    "Age_yrs",
    "Floor",
    "Total_Floors",
    "Dist_Rail_km",
    "Sea_View",
    "Parking",
    "Price_INR"
]

numeric_cols = [c for c in numeric_cols if c in df.columns]
corr = df[numeric_cols].corr()

print("\nCorrelation matrix (numeric features):")
print(corr)


# In[21]:


df["Price_Lakh"] = df["Price_INR"] / 1e5


# In[23]:


if "Locality" in df.columns:
    df_encoded = pd.get_dummies(df,columns=["Locality"], drop_first=True)
else:
    df_encoded = df.copy()


# In[24]:


df=pd.read_csv("mumbai_housing_data.csv")
print("First 5 rows of raw data:")
print(df_encoded.head())


# In[25]:


target = "Price_Lakh"
feature_cols = [col for col in df_encoded.columns if col not in ["Price_INR",target]]


# In[26]:


x = df_encoded[feature_cols]
y= df_encoded[target]
x_train, x_test, y_train , y_test = train_test_split(x,y, test_size =0.2, random_state=42)


# In[27]:


print(f"\nTraining samples:{x_train.shape[0]},Test samples:{x_test.shape[0]}")


# In[29]:


model = LinearRegression()
model.fit(x_train, y_train)
print("\nModel trained:Linear Regression")
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
print("\nEvaluation Metrics(Test Set):")
print(f"Mean Absolute Error (Mae): {mae:.2f} Lakh")
print(f"Root Mean Squared Error (rmse):{rmse:.2f}Lakh")
print(f"R^2 Score:{r2:.4f}")


# In[30]:


plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Price (Lakh)")

plt.ylabel("Predicted Price (Lakh)") 

plt.title("Actual vs Predicted Prices (Test Set)") 

min_val = min(y_test.min(), y_pred.min()) 

max_val = max(y_test.max(), y_pred.max()) 

plt.plot([min_val, max_val], [min_val, max_val], "r--") 

plt.tight_layout() 

plt.show()
 


# In[31]:


results = pd.DataFrame({ 
"Actual_Lakh": y_test.values[:10], 
"Predicted_Lakh": y_pred[:10] }) 
print("\nSample prediction comparison (first 10 test rows):")
print(results.round(2))


# In[ ]:




