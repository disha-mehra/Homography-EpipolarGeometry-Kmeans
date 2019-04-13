import cv2
import numpy as np


UBIT = "dishameh"
np.random.seed(sum([ord(c) for
c in UBIT]))

img1 = cv2.imread('tsucuba_left.png')  #Left
img2 = cv2.imread('tsucuba_right.png') #Right

sift = cv2.xfeatures2d.SIFT_create()

#Keypoints and descriptors with SIFT
keypoint_1, descriptor_1 = sift.detectAndCompute(img1,None)
keypoint_2, descriptor_2 = sift.detectAndCompute(img2,None)

img3=cv2.drawKeypoints(img1,keypoint_1,None)
cv2.imwrite('task2_sift1.jpg',img3)

img4=cv2.drawKeypoints(img2,keypoint_2,None)
cv2.imwrite('task2_sift2.jpg',img4)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptor_1,descriptor_2,k=2)

good_1 = []
good_2= []

# ratio test as per Lowe's paper
for (m,n) in matches:
    if m.distance < 0.75*n.distance:
       good_1.append([m])
       good_2.append(m)

img5 = cv2.drawMatchesKnn(img1,keypoint_1,img2,keypoint_2,good_1,None,flags=2)
cv2.imwrite('task2_matches_knn.jpg',img5)

#Task 1.3

src_pts = np.float32([ keypoint_1[m.queryIdx].pt for m in good_2 ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoint_2[m.trainIdx].pt for m in good_2 ]).reshape(-1,1,2)

pts1=np.int32(src_pts)
pts2=np.int32(dst_pts)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
print("Fundamental Matrix:\n", F)

# We select only inlier points
pts_1 = pts1[mask.ravel()==1]
pts_2 = pts2[mask.ravel()==1]   

ind_random=np.random.choice(good_2,10)
##print(ind_random)
#

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape[:2]
    count=0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = color_line[count]
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
        count=count+1
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
color_line=[]
for i in range(0,10):
    color_line.append(tuple(np.random.randint(0,255,3).tolist()))

lines1 = cv2.computeCorrespondEpilines(pts_2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts_1[:10],pts_2[:10])

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts_1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts_2[:10],pts_1[:10])

cv2.imwrite('task2_epi_right.jpg',img5)
cv2.imwrite('task2_epi_left.jpg',img3)

#Task 1.4
stereoMatcher = cv2.StereoBM_create(numDisparities=80, blockSize=15)    

imgL=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
imgR=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

disparity_img = stereoMatcher.compute(imgL,imgR)

largest_num = disparity_img[0][0]
for row_idx, row in enumerate(disparity_img):
    for col_idx, num in enumerate(row):
        if num > largest_num:
            largest_num = num
large_val = largest_num

row = disparity_img.shape[0]
col = disparity_img.shape[1]
for i in range(row):
    for j in range(col):
        disparity_img[i][j] = (disparity_img[i][j]/large_val)*255

disparity_normalize=disparity_img

cv2.imwrite('task2_disparity.jpg',disparity_normalize)

