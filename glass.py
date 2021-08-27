# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# %%
glass = pd.read_csv("glass.csv")
glass


# %%
glass.info()


# %%
glass.shape


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## **Data Distribution before normalization**

# %%
a = 3  # number of rows
b = 4  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(20,15))

for i in glass.columns:
    plt.subplot(a, b, c)
    plt.title(' Data Distribution'.format(i))

    sns.histplot(x= i ,data= glass )

    # plt.title(f" Gender vs {i}")

    c = c + 1

plt.show()


# %%

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


# %%
# Normalized data frame (considering the numerical part of data)
glass_n = norm_func(glass.iloc[:, :])
glass_n.describe()


# %%
glass_n.info()


# %%
X = np.array(glass_n.iloc[:,:]) # Predictors 
Y = np.array(glass['Type']) # Target 


# %%
from sklearn.model_selection import train_test_split


# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# %%
from sklearn.neighbors import KNeighborsClassifier


# %%
knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)


# %%
pred = knn.predict(X_test)
pred


# %%
# Evaluate the model
from sklearn.metrics import accuracy_score


# %%
print(accuracy_score(Y_test, pred))


# %%
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# %%
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))


# %%
# error on train data
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# %%
# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


# %%
import matplotlib.pyplot as plt # library to do visualizations 


# %%
# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
plt.show()


# %%
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
plt.show()


# %%



