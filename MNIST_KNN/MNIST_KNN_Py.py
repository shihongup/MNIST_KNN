
# coding: utf-8

# <h1><font size="6">CSCI 6364 - Machine Learning</font></h1>
# <h1><font size="5">Project 1 - MNIST</font></h1>
# <p><font size="4"><span style="line-height:30px;">Student: Shifeng Yuan</span><br>
# <span style="line-height:30px;">GWid: G32115270<span><br>
# Language: Python<font><br>
# <span style="line-height:30px;">Resource: MNIST data from Kaggle <span></p>

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
#get_ipython().run_line_magic('matplotlib', 'inline')


# <h1>1. Dataset Details</h1>
# <p> &nbsp; <font size="3">Here we have two datasets, one is training data and another is the testing data.</font></p>
# <ol><font size="3">
# <li>The training data contains the data of 28000 images</li>
# <li>Each image is described as 28*28=784 columns with numbers representing its lightness or darkness</li>
# <li>The first column is the actual number of what the image represents</li>
# <li>The testing data is the same as the training data but it does not have the "label" column which should be generated.</li>
# <font></ol>

# <h2>Data Spliting</h2>
# <p><font size="3">First, we need to read the data into a variable called dataset. And we should split the data into images and labels as two parts. Usually, we divide our dataset into 2 to 3 parts. Here, I split the dataset into training data (80%)  and testing data(20%)</font></p>

# In[5]:


dataset=pd.read_csv('data/mnistdata/train.csv')
images=dataset.iloc[0:28000,1:]
labels=dataset.iloc[0:28000,:1]
train_images,test_images,train_labels,test_labels=train_test_split(images,labels,random_state=2,test_size=0.2)


# <h2> Inspect the Dataset </h2>

# In[93]:


# Read the dataset and then print the head
print( len(dataset) )
print( dataset.head() )


# <h2>Dataset Visualization</h2>
# <p><font size="3">We can use the .imshow() in the matplotlib package to visualize the data as a picture. We first select a row, here it the 4th row, and then reshape it into 28*28 matrix, finally the package gives the picture.</font></p>

# In[61]:


i=3
img=images.iloc[i].values
img=img.reshape(28,28)
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i])


# <p><font size="3">Use the .hist() to draw a histgram of the data.</font></p>

# In[68]:


plt.hist(images.iloc[0])


# <h1>2. Algorithm Description</h1>
# <p><font size="3"></font></p>

# <h2>Selection of K</h2>
# <p><font size="3">The selection of value K is important for KNN, usually, we make K the square root of the size of the test sample, however, because this dataset is too big, we simply make it 5.</font></p>

# <p><font size="3">The package sklearn.neighbors.KNeighborsClassifier implementing the K-nearest Neighbors
# classification.</font></p>
# <p><font size="3">
# Using the sklearn KNeighborsClassifier package, define the metric method as euclidean.
# we simply use a brute force algorithm.</font></p>

# In[13]:


from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute', p = 2, metric = 'euclidean')
clf.fit(train_images,train_labels.values.ravel())


# <h1>3. Algorithm Results</h1>
# <p><font size="3">Start predict and measure the accuracy of the algorithm.</font></p>

# In[14]:


predictions=clf.predict(test_images)


# In[15]:


print(predictions)


# <h2>Confusion Matrix</h2>

# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, predictions)
print(cm)


# <p><font size="3">The full confusion matrix shown below, and the accuracy score is 0.9578571428571429.</font></p>

# <img src='../MNIST_confusion.png' width = '800' height='500'>

# In[46]:


from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,predictions))


# <p><font size="3">Then start predicting the test data in the test.csv</font></p>

# In[8]:


# read the test data into variable testd
testd=pd.read_csv('data/mnistdata/test.csv')


# In[47]:


result=clf.predict(testd)


# In[48]:


print(result)


# <p><font size="3">Choosing the 100th number in the test set so see the variance caused by the K.</font></p>

# In[9]:


img_100 = testd.iloc[99:100,:]


# In[86]:


# k=5
result1 = clf.predict(img_100)
print(result1)


# In[87]:


# k=9
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors = 9, algorithm = 'brute', p = 2, metric = 'euclidean')
clf.fit(train_images,train_labels.values.ravel())
result2 = clf.predict(img_100)
print(result2)


# In[88]:


# k=3
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors = 3, algorithm = 'brute', p = 2, metric = 'euclidean')
clf.fit(train_images,train_labels.values.ravel())
result3 = clf.predict(img_100)
print(result3)


# In[89]:


# k=11
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors = 9, algorithm = 'brute', p = 2, metric = 'euclidean')
clf.fit(train_images,train_labels.values.ravel())
result4 = clf.predict(img_100)
print(result4)


# <p><font size="3">It turns out that the 100th number is predicted as 4 when k=5,9,3,11. The accuracy is fine.</font></p>

# In[50]:


# Output the result as .csv file
df=pd.DataFrame(result)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv',header=True)


# <h1>4. Runtime</h1>
# <p><font size="3"></font></p>

# <p><font size="3">
#     For d dimension, we need O(d) runtime to compute one distance between two data, so computing all the distance between one data to other data needs O(nd) runtime, then we need O(kn) runtime to find the K nearest neibors, so, in total, it takes O(dn+kn) runtime for the classifier to classify the data.
# </font></p>

# In[10]:


import time
start = time.time()
clf=KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute', p = 2, metric = 'euclidean')
clf.fit(train_images,train_labels.values.ravel())
result=clf.predict(testd)
end = time.time()
print(end-start)


# <p><font size="3">
#     As is shown above, the "wall-clock" of the runtime is about 158.66s
# </font></p>
