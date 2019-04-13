
# coding: utf-8

# # Task 3- K-means Clustering

# ## Ans 1

# Here we have included the basic libraries

# In[99]:


import numpy as np
from numpy import reshape
import cv2
from matplotlib import pyplot as plt

UBIT = "dishameh"
np.random.seed(sum([ord(c) for
c in UBIT]))


# Put the data points specified in the question in different lists that is list specifying the x and y axis values of the different data points. We specify the lists of the 3 centroids as mentioned as well.  

# In[100]:


data_point_x=[5.9,4.6,6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0]
data_point_y=[3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0]
centroids=[[6.2,3.2],[6.6,3.7],[6.5,3.0]]
points=np.array(list(zip(data_point_x,data_point_y)))
# print("Data points are:", data_point_x[])
centroid = np.array(centroids)
# print("Centroids are:",centroid)


# We plot the data points along with their centroids as in the question. 

# In[101]:


plt.scatter(data_point_x, data_point_y, c='white', s=100, marker="^"
            ,edgecolors='blue')
for i in range(len(data_point_x)):
    plt.text(data_point_x[i]-0.12, data_point_y[i]-0.12,
             "("+str(data_point_x[i])+","+str(data_point_y[i])+")", fontsize=8)

colors=['red','green','blue']
for i in range(0,3):
    plt.scatter(centroids[i][0],centroids[i][1], c=colors[i],s=100,marker="o"
                ,edgecolors=colors[i])
    plt.text(centroids[i][0]-0.12, centroids[i][1]-0.12,
             "("+str(centroid[i][0])+","+str(centroid[i][1])+")", fontsize=8)
plt.show()


# In[102]:


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# In[106]:


# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(points))
print("Initial Clusters:",clusters)
# Assigning each value to its closest cluster
for i in range(len(points)):
    distances = dist(points[i], centroid)
#     print("Distances:", distances)
    cluster = np.argmin(distances)
    clusters[i] = cluster
print("Classification Vector:",clusters)


# In[109]:


colors = ['red', 'green', 'blue']
# fig, ax = plt.subplots()
for i in range(len(points)):
    if clusters[i]==0:
        plt.scatter(points[i][0],points[i][1], c=colors[0],s=100,marker="^"
                    ,edgecolors=colors[0])
        plt.text(points[i][0]-0.12, points[i][1]-0.12,
             "("+str(points[i][0])+","+str(points[i][1])+")", fontsize=8)

    elif clusters[i]==1:
        plt.scatter(points[i][0],points[i][1], c=colors[1],s=100,marker="^"
                    ,edgecolors=colors[1])
        plt.text(points[i][0]-0.12, points[i][1]-0.12,
             "("+str(points[i][0])+","+str(points[i][1])+")", fontsize=8)

    else:
        plt.scatter(points[i][0],points[i][1], c=colors[2],s=100,marker="^"
                    ,edgecolors=colors[2])
        plt.text(points[i][0]-0.12, points[i][1]-0.12,
             "("+str(points[i][0])+","+str(points[i][1])+")", fontsize=8)


        for i in range(0,3):
            plt.scatter(centroids[i][0],centroids[i][1], c=colors[i],s=100,marker="o"
                ,edgecolors=colors[i])
            plt.text(centroids[i][0]-0.12, centroids[i][1]-0.12,
             "("+str(centroid[i][0])+","+str(centroid[i][1])+")", fontsize=8)

plt.savefig('task3_iter1_a.jpg',dpi=500)


# ## Ans 2

# In[110]:


new_centroids = np.zeros(centroid.shape)
print(new_centroids)
# Finding the new centroids by taking the average value
colors = ['red', 'green', 'blue']
for i in range(3):
        data_points = [points[j] for j in range(len(points)) if clusters[j] == i]
        new_centroids[i] = np.mean(data_points, axis=0)
for i in range(0,3):
    plt.scatter(new_centroids[i][0],new_centroids[i][1], c=colors[i],s=100,marker="o"
                ,edgecolors=colors[i])
    plt.text(new_centroids[i][0]-0.07, new_centroids[i][1]-0.12,
             "("+str(np.float16(new_centroids[i][0]))+","+str(np.float16(new_centroids[i][1]))+")", fontsize=8)
print("New centroids after one iteration", new_centroids)

plt.savefig('task3_iter1_b.jpg',dpi=500)

# Cluster Lables(0, 1, 2)
print("Clusters:")
new_clusters = np.zeros(len(points))
print("Initial Clusters:",clusters)
# Assigning each value to its closest cluster
for i in range(len(points)):
    distances = dist(points[i], new_centroids)
#     print("Distances:", distances)
    new_cluster = np.argmin(distances)
    new_clusters[i] = new_cluster
print("Classification Vector:",new_clusters)


# ## Ans 3

# In[111]:


new_centroids_1 = np.zeros(centroid.shape)
print(new_centroids_1)
# Finding the new centroids by taking the average value
colors = ['red', 'green', 'blue']
for i in range(3):
        data_points = [points[j] for j in range(len(points)) if new_clusters[j] == i]
        new_centroids_1[i] = np.mean(data_points, axis=0)
for i in range(0,3):
    plt.scatter(new_centroids_1[i][0],new_centroids_1[i][1], c=colors[i],s=100,marker="o"
                ,edgecolors=colors[i])
    plt.text(new_centroids_1[i][0]-0.07, new_centroids_1[i][1]-0.08,
             "("+str(np.float32(new_centroids_1[i][0]))+","+str(np.float32(new_centroids_1[i][1]))+")", fontsize=8)
print("New centroids after second iteration", new_centroids_1)

plt.savefig('task3_iter2_a.jpg',dpi=500)


# Cluster Lables(0, 1, 2)
print("New Clusters")
new_clusters_1 = np.zeros(len(points))
print("Initial Clusters:",new_clusters)
# Assigning each value to its closest cluster
for i in range(len(points)):
    distances = dist(points[i], new_centroids_1)
#     print("Distances:", distances)
    new_cluster_1 = np.argmin(distances)
    new_clusters_1[i] = new_cluster_1
print("Classification Vector:",new_clusters_1)


# In[112]:


colors = ['red', 'green', 'blue']
# fig, ax = plt.subplots()
for i in range(len(points)):
    if new_clusters_1[i]==0:
        plt.scatter(points[i][0],points[i][1], c=colors[0],s=100,marker="^"
                    ,edgecolors=colors[0])
        plt.text(points[i][0]-0.12, points[i][1]-0.12,
             "("+str(points[i][0])+","+str(points[i][1])+")", fontsize=8)

    elif new_clusters_1[i]==1:
        plt.scatter(points[i][0],points[i][1], c=colors[1],s=100,marker="^"
                    ,edgecolors=colors[1])
        plt.text(points[i][0]-0.12, points[i][1]-0.12,
             "("+str(points[i][0])+","+str(points[i][1])+")", fontsize=8)

    else:
        plt.scatter(points[i][0],points[i][1], c=colors[2],s=100,marker="^"
                    ,edgecolors=colors[2])
        plt.text(points[i][0]-0.12, points[i][1]-0.12,
             "("+str(points[i][0])+","+str(points[i][1])+")", fontsize=8)

for i in range(0,3):
    plt.scatter(new_centroids_1[i][0],new_centroids_1[i][1], c=colors[i],s=100,marker="o"
                ,edgecolors=colors[i])
    plt.text(new_centroids_1[i][0]-0.12, new_centroids_1[i][1]-0.12,
             "("+str(np.float32(new_centroids_1[i][0]))+","+str(np.float32(new_centroids_1[i][1]))+")", fontsize=8)

    
plt.savefig('task3_iter2_b.jpg',dpi=500)


# ## Task 4

# In[168]:


img = cv2.imread('baboon.jpg')


# In[169]:


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(img.shape)
pixel = img.reshape((img.shape[0]*img.shape[1],3))
# print("Reshaped Pixel")
# print(pixel)
# print(pixel[262143][0])


# In[170]:


point_x=[]
point_y=[]
point_z=[]
for i in range(0,262144):
    point_x.append(pixel[i][0]) 
    point_y.append(pixel[i][1])
    point_z.append(pixel[i][2])
points = np.array(list(zip(point_x, point_y,point_z)))


# In[171]:


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# ## k=3

# In[117]:


# Number of clusters
k =3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(points)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
C_z = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)

# print(C)


# In[118]:


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(points))
#print(clusters.shape)
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# error=1
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(points)):
        distances = dist(points[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = C.copy()
#     print(clusters)
    # Finding the new centroids by taking the average value
    for i in range(k):
        n_points = [points[j] for j in range(len(points)) if clusters[j] == i]
        C[i] = np.nanmean(n_points, axis=0)
#         print(C)
    error = dist(C, C_old, None)
#     error =error-1


# In[119]:


labels = np.uint8(clusters)
# print(labels)
centers = np.uint8(C)
# print(centers)
less_colors = centers[labels].reshape(img.shape).astype('uint8')
cv2.imshow('image',less_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('task3_baboon_3.jpg',less_colors)


# ## k=5

# In[120]:


# Number of clusters
k =5
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(points)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
C_z = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)

# print(C)


# In[121]:


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(points))
#print(clusters.shape)
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# error=1
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(points)):
        distances = dist(points[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = C.copy()
#     print(clusters)
    # Finding the new centroids by taking the average value
    for i in range(k):
        n_points = [points[j] for j in range(len(points)) if clusters[j] == i]
        C[i] = np.nanmean(n_points, axis=0)
#         print(C)
    error = dist(C, C_old, None)
#     error =error-1


# In[122]:


labels = np.uint8(clusters)
# print(labels)
centers = np.uint8(C)
# print(centers)
less_colors_5 = centers[labels].reshape(img.shape).astype('uint8')
cv2.imwrite('task3_baboon_5.jpg',less_colors_5)


# ## k=10

# In[125]:


# Number of clusters
k =10
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(points)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
C_z = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)

# print(C)


# In[126]:


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(points))
#print(clusters.shape)
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# error=1
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(points)):
        distances = dist(points[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = C.copy()
#     print(clusters)
    # Finding the new centroids by taking the average value
    for i in range(k):
        n_points = [points[j] for j in range(len(points)) if clusters[j] == i]
        C[i] = np.nanmean(n_points, axis=0)
#         print(C)
    error = dist(C, C_old, None)
#     error =error-1


# In[127]:


labels = np.uint8(clusters)
# print(labels)
centers = np.uint8(C)
# print(centers)
less_colors_10 = centers[labels].reshape(img.shape).astype('uint8')
cv2.imwrite('task3_baboon_10.jpg',less_colors_10)


# ## k=20

# In[175]:


# Number of clusters
k =20
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(points)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
C_z = np.random.randint(0, np.max(points)-20, size=k)
C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)

# print(C)


# In[ ]:


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(points))
#print(clusters.shape)
# Error func. - Distance between new centroids and old centroids
# error = dist(C, C_old, None)
error=1
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(points)):
        distances = dist(points[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = C.copy()
#     print(clusters)
    # Finding the new centroids by taking the average value
    for i in range(k):
        n_points = [points[j] for j in range(len(points)) if clusters[j] == i]
        C[i] = np.nanmean(n_points, axis=0)
#         print(C)
#     error = dist(C, C_old, None)
    error =error-1
    


# In[163]:


labels = np.uint8(clusters)
# print(labels)
centers = np.uint8(C)
# print(centers)
less_colors_20 = centers[labels].reshape(img.shape).astype('uint8')
cv2.imwrite('task3_baboon_20.jpg',less_colors_20)

