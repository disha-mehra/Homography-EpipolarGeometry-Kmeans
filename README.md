# Homography-EpipolarGeometry-Kmeans
Performing Homography, Epipolar Geometry and K-means clustering to image color quantization

**University at Buffalo - CSE573: Computer Vision and Image Processing**
<p>Project 2</p>

### Overview
* Image Features and Homography
  * Given two images, extract SIFT features and draw keypoints for both images.
  * Match the keypoints using k-nearest neighbor
  * Wrap the first image to the secong image using Homography matrix
  
* Epipolar Geometry
  * Given two images, extract SIFT features and draw keypoints for both images. 
  * Compute fundamental matrix with RANSAC and draw epilines.
  * Compute disparity map and show disparity image
  
* K-means Clustering
  * Apply k-means to image color quantization. Using only k colors to represent the image baboon.jpg.
  
### Software Used
Python

### Visuals
* Panorama Image obtained using Homography
![pano](https://github.com/disha-mehra/Homography-EpipolarGeometry-Kmeans/blob/master/Outputs/task1_pano.jpg)

* Disparity Image obtained
![disparity](https://github.com/disha-mehra/Homography-EpipolarGeometry-Kmeans/blob/master/Outputs/task2_disparity.jpg)

* K-means clustering for color quantization
![kmeans](https://github.com/disha-mehra/Homography-EpipolarGeometry-Kmeans/blob/master/Outputs/task3_baboon_20.jpg)
