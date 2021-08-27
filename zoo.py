# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# %%
zoo = pd.read_csv("Zoo.csv")
zoo


# %%
zoo.info()


# %%
categorical_features=[feature for feature in zoo.columns if zoo[feature].dtypes=='O']
categorical_features


# %%
for var in categorical_features:
    
    print(var, ' contains ', len(zoo[var].unique()), ' labels')


# %%
zoo.shape


# %%
y=zoo['type'].values
X=zoo.drop(['type','animal name'],axis=1).values


# %%
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# %%
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


# %%
knn.score(X_train,y_train)


# %%
knn.score(X_test,y_test)


# %%
pred = knn.predict(X_test)
pred


# %%
# Evaluate the model
from sklearn.metrics import accuracy_score


# %%
print(accuracy_score(y_test, pred))


# %%
pd.crosstab(y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# %%
pred_train = knn.predict(X_train)
print(accuracy_score(y_train, pred_train))


# %%
# error on train data
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# %%
# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
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



