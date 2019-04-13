# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:56:21 2018

@author: disha
"""
import numpy as np
import cv2

UBIT = "dishameh"
np.random.seed(sum([ord(c) for
c in UBIT]))

#Task 1.1

img1 = cv2.imread('mountain1.jpg')
img2 = cv2.imread('mountain2.jpg')

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
keypoint_1, descriptor_1 = sift.detectAndCompute(img1,None)
keypoint_2, descriptor_2 = sift.detectAndCompute(img2,None)

img3=cv2.drawKeypoints(img1,keypoint_1,None)
cv2.imwrite('task1_sift1.jpg',img3)

img4=cv2.drawKeypoints(img2,keypoint_2,None)
cv2.imwrite('task1_sift2.jpg',img4)

#Task 1.2

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=10)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(descriptor_1,descriptor_2,k=2)

# ratio test as per Lowe's paper
good_1=[]
good_2=[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
#        matchesMask[i]=[1,0]
        good_1.append([m])
        good_2.append(m)


img5 = cv2.drawMatchesKnn(img1,keypoint_1,img2,keypoint_2,good_1,None,flags=2)
cv2.imwrite('task1_matches_knn.jpg',img5)

#Task 1.3

src_pts = np.float32([ keypoint_1[m.queryIdx].pt for m in good_2 ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoint_2[m.trainIdx].pt for m in good_2 ]).reshape(-1,1,2)
#print(src_pts)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

print("Homography Matrix:",H)
matchesMask = mask.ravel().tolist()


#Task 1.4

ind_random=np.random.choice(good_2,10)

random_mask=np.random.choice(matchesMask,10)
print(ind_random[0])


draw_params = dict(matchColor=(
    0, 0, 255), singlePointColor=None, matchesMask=random_mask.tolist(), flags=2)


img6=cv2.drawMatches(img1,keypoint_1,img2,keypoint_2,ind_random,None,**draw_params)
cv2.imwrite('task1_matches.jpg',img6)


#Task 1.5

rows1, cols1 = img1.shape[:2]
rows2, cols2 = img2.shape[:2]

list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

translation_dist = [-x_min, -y_min]
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

img7 = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
img7[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img2

cv2.imwrite('task1_pano.jpg',img7)


