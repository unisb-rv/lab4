# Feature Detection (edge, corners, Hough transformation)

## Edge Detection

When talking about images, the edge can be defined as a boundary between two regions with relatively distinct gray level properties.
Edges are pixels where the brightness function changes abruptly.
Edge detectors are a collection of very important local image preprocessing methods used to locate (sharpen) changes in the intensity function.
Different edge detection methods include Canny, Sobel, Roberts, SUSAN, Prewitt, and Deriche.

### Edge Detection with Canny Edge descriptor 

Canny Edge Detection is a popular edge detection algorithm. 
This is a multi-step algorithm, so its successful performance, depends on few steps: 

<ul>
  <li><b>Preprocessing (noise reduction)</b></li>
  Since all edge detection results are easily affected by image noise, it is essential to filter out the noise to prevent false detection caused by noise. 
  To smooth the image, we will use a Gaussian filter. 
  This step will slightly smooth the image to reduce the effects of obvious noise on the edge detector.
  The equation for a Gaussian filter with a kernel of size (2k+1) x (2k+1) is given with: 

![gauss](https://wikimedia.org/api/rest_v1/media/math/render/svg/4a36d7f727beeaff58352d671bb41a3aca9f44d6)
  
  As seen from the above formula, important parameters for the Gaussian filter are:
  <ul>
    <li>size of the kernel (mostly 5x5 kernel is used)</li>
    <li>and standard deviation sigma </li>
  </ul>
  
   <p>
  <li><b>Finding intensity gradient of the image</b></li>
  An edge in an image may point in a variety of directions, so the Canny algorithm uses filters to detect horizontal, vertical and diagonal edges in the blurred image. 
  The edge detection operator (such as Roberts, Prewitt, or Sobel) returns a value for the first derivative in the horizontal direction (Gx) and the vertical direction (Gy). 
  In other words, <i>the magnitude</i> of the gradient at a point in the image determines if it possibly lies on the edge or not. A high gradient magnitude means the colors are changing rapidly which implies the existence of an edge while a low gradient implies that edge is not present. 
  The <i>direction</i> of the gradient shows how the edge is oriented.
  To calculate these, following formulas are used: 

![canny-edge-1](http://latex.codecogs.com/gif.latex?Edge%20Gradient%20%28G%29%20%3D%20%5Csqrt%7BG_x%5E2&amp;plus;G_y%5E2%7D)
![canny-edge-2](http://latex.codecogs.com/gif.latex?Angle%20%28%5Ctheta%29%20%3D%20%5Ctan%5E%7B-1%7D%5Cfrac%7BG_y%7D%7BG_x%7D)

  Once we have the gradient magnitudes and orientations, we can get started with the actual edge detection.
 
 </p>

 <p>
  <li><b>Applying non-maximum suppression to get rid of spurious response to the edge detection</b></li>

  After gradient magnitude and direction are obtained, a full scan of the image is done to remove any unwanted pixels which may not constitute the edge.
  Therefore, edge thining technique known as non-maximum suppression is applied to find all those unwanted pixels. 
  For this, at every pixel, the pixel is checked if it is a local maximum in its neighborhood in the direction of the gradient. Check the image below: 

![non-maximum](https://docs.opencv.org/3.1.0/nms.jpg)
  <br>
  Point A is on the edge ( in the vertical direction). Gradient direction is normal to the edge. Point B and C are in gradient directions. 
  So point A is checked with point B and C to see if it forms a local maximum. If so, it is considered for the next stage, otherwise, it is suppressed ( put to zero).
  In short, the result you get is a binary image with "thin edges".
</p>

  <li><b>Track edge by hysteresis </li></b>
  Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
  This stage decides which are all edges are really edges and which are not. For this, we need two threshold values, minVal, and maxVal. 
  Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges, so discarded. 
  Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. 
  If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded. See the image below:

![non-maximum](https://docs.opencv.org/3.1.0/hysteresis.jpg)

 <p>
    The edge A is above the maxVal, so considered as "sure-edge". Although edge C is below maxVal, it is connected to edge A, so that also considered as valid edge and we get that full curve. 
    But edge B, although it is above minVal and is in the same region as that of edge C, it is not connected to any "sure-edge", so that is discarded. 
    So it is very important that we have to select minVal and maxVal accordingly to get the correct result. 
    This stage also removes small pixels noises on the assumption that edges are long lines. So what we finally get is strong edges in the image.
 </p>
  
</ul>

#### OpenCV implementation of Canny edge detector:

```
Function : cv2.Canny(blurred image,lower threshold,upper threshold)
Parameters are as follows :
1. blurred image : input image blurred with Gaussian 5 by 5 kernel
2. lower threshold : first threshold for the hysteresis procedure
3. upper threshold : second threshold for the hysteresis procedure
```
More information can be found at:  <a href="https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html?highlight=cv2.canny#cv2.Canny">OpenCV cv2.Canny documentation</a> 

## Task 1

Try the Canny edge detector on an arbitrary image within the `slike` directory. **Load the image as a grayscale image.** Also, before applying Canny edge detection, blur the image using **Gaussian blur**. (Refer to previous exercises.) Display the original image and the edge-detected image. Visually determine the values of the lower and upper thresholds to obtain clear edges without too many extra ones. Good starting values are between 100 and 200.

As you could observe, the common problem here is in determining these lower and upper thresholds. So, what is the optimal value for the thresholds?
This question is especially important when you are processing multiple images with different contents captured under varying lighting conditions.  
A little trick that relies on basic statistics can help you automatically determ these values, removing the manual tuning of the thresholds for Canny edge detection.

## Hough Transformation for Lines

When images are to be used in different areas of image analysis such as object recognition, it is important to reduce the amount of data in the image while preserving the important,
characteristic, structural information. Edge detection makes it possible to reduce the amount of data in an image considerably. However the output from an edge detector is still a image
described by it’s pixels. If lines, ellipses and so forth could be defined by their characteristic equations, the amount of data would be reduced even more. 
The Hough transform was originally developed to recognize lines and has later been generalized to cover arbitrary shapes. 

<ul>
<li> <b>Representation of Lines in the Hough Space</b> </li>

Lines can be represented uniquely by with parameters a and b and following equation: 

![line-1](https://latex.codecogs.com/gif.latex?y%3Da%5Ccdot%20x%20&plus;b)

Above equation is not able to represent vertical lines. Therefore, the Hough transform uses the following equation: <br>

![line-2](https://latex.codecogs.com/gif.latex?r%20%3D%20x%20%5Ccdot%20%5Ccos%20%5CTheta%20&plus;%20y%5Ccdot%20%5Csin%20%5CTheta)

To obtain similar equation to the first one, this can be rewritten as: 

![line-3](https://latex.codecogs.com/gif.latex?y%20%3D%20-%20%5Cfrac%7B%5Ccos%20%5CTheta%20%7D%7B%5Csin%20%5CTheta%20%7D%20%5Ccdot%20x%20&plus;%20%5Cfrac%7Br%7D%7B%5CTheta%20%7D)

The parameters ![line-1](https://latex.codecogs.com/gif.latex?%5Ctheta) and r is the angle of the line and the distance from the line to the origin, respectively. 
All lines can be represented in this form when  <img src="https://i.ibb.co/4Sq4DkD/formula1.png" alt="hough1" border="0"></a>.
To sum up, the Hough space for lines has these two dimensions: ![line-1](https://latex.codecogs.com/gif.latex?%5Ctheta) and r and a line is represented
by a single point, corresponding to a unique set of parameters ![line-1](https://latex.codecogs.com/gif.latex?%28%5Ctheta%20_%7B0%7D%2Cr_%7B0%7D%29) . 
The line-to-point mapping is illustrated in the following image: 
<p align="center">
<img src="https://i.ibb.co/gJ9C8Jy/hough1.png" alt="hough1" border="0"></a><br /><br />
</p>

<li> <b>Mapping of edge points to the Hough space</b> </li>

An important concept for the Hough transform is the mapping of single points. The idea is, that a point is mapped to all lines, that can pass through that point.
This yields a sine-like line in the Hough space.  This principle is illustrated for a point ![formula](https://latex.codecogs.com/gif.latex?p_%7B0%7D%3D%20%2840%2C30%29)as shown in following figures: 
<p align="center">
<img src="https://i.ibb.co/GVRDyJs/houghspace.png" alt="houghspace" border="0">
</p>
On the left image transformation of a single point to a line in the Hough space is shown while on the right image 
the Hough space line representation through all possible lines through the point is shown. 

#### The Hough Space Accumulator

To determine the areas where most Hough space lines intersect, an accumulator covering the Hough space is used. When an edge point is transformed, bins in the accumulator is incremented
for all lines that could pass through that point. The resolution of the accumulator determines the precision with which lines can be detected. 
In general, the number of dimensions of the accumulator corresponds to the number of unknown parameters in the Hough transform problem. Thus, for ellipses a 5-dimensional space is required
(the coordinates of its center, the length of its major and minor axis, and its angle). For lines 2 dimensions suffice (r and θ). This is why it is possible to visualize the content of the ac
cumulator.

To sum up, the algorithm for detecting straight lines can be divided to the following steps: 
<ul>
<li> Edge detection, e.g. using the Canny edge detector </li>
<li> Mapping of edge points to the Hough space and storage in an accumulator </li>
<li> Interpretation of the accumulator to yield lines of infinite length. The interpretation isdone by thresholding and possibly other constraints. </li>
<li> Conversion of infinite lines to finite lines. </li>
</ul>

More information about Hough Transformation for lines can be found at:  <a href="href> https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html">OpenCV Hough Transform documentation</a> 

OpenCV implementation of Hough Transform for lines can be found in: 

```
Function : cv2.HoughLines(image with edges,rho,theta,threshold, array,srn,stn)
Parameters are as follows :
1. image with edges : input image with found edges
2. rho : distance resolution of the accumulator in pixels
3. theta : angle resolution of the accumulator in radians
4. threshold: accumulator threshold parameter. only those lines are returned that get enough votes 
5. array: return an empty array with shape and type of input for storing result
6. srn: for the multi-scale Hough transform, if 0 standard Hough transform is used
7. stn: for the multi-scale Hough transform, if 0 standard Hough transform is used
```
More information can be found at:  <a href="https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html?highlight=cv2.hough#cv2.HoughLines">OpenCV cv2.HoughLines documentation</a> 

## Task 2

On the crossword.jpg image, perform the Hough transformation. Since the input to the HoughLines function is an edge-detected image, you first need to perform Canny edge detection as in the first task, including the blur step. Then, perform the Hough transformation on the edge image. After completing the transformation, call the draw_lines function to draw the detected lines on the original image. Display the resulting image. Example call:

`lines = cv2.HoughLines(edges, 1, math.pi/90, 200, np.array([]), 0, 0)`

These are good starting values for the parameters. Adjust the threshold parameter, which is set to 200 in the example call, to get the lines of the crossword.

## Corner Detection

Corners are locations in images where a slight shift in the location will lead to a large change in intensity in both horizontal and vertical axes.
The Harris Corner detection algorithm consist of following steps:

<ul>
<li><b>Determination of windows (small image patches) with large intensity variation</b></li>

Let a window (the center) be located at position  &nbsp; ![cornerr](https://latex.codecogs.com/gif.latex?%28x%2Cy%29).  &nbsp; 
Let the intensity of the pixel at this location be  &nbsp; ![cornerrr](https://latex.codecogs.com/gif.latex?I%28x%2Cy%29).  &nbsp;
If this window slightly shifts to a new location with displacement  &nbsp; ![corner7](https://latex.codecogs.com/gif.latex?%28u%2Cv%29),  &nbsp; the intensity of the pixel at this location will be 
 &nbsp; ![corner8](https://latex.codecogs.com/gif.latex?I%28x&plus;u%2Cy&plus;v%29).  &nbsp; 

Therefore,  &nbsp; ![corner9](https://latex.codecogs.com/gif.latex?%5BI%28x&plus;u%2Cy&plus;v%29-I%28x%2Cy%29%5D)  &nbsp; will be the difference in intensities of the window shift. 
For a corner, this difference will be very high. 
We maximize this term by differentiating it with respect to the X and Y axes. 
Let  &nbsp; ![corner10](https://latex.codecogs.com/gif.latex?w%28x%2Cy%29)  &nbsp;be the weights of pixels over a window (Rectangular or a Gaussian).
Then,  &nbsp; ![corner11](https://latex.codecogs.com/gif.latex?E%28u%2Cv%29)  &nbsp; is defined as :
<p align="center">
<img src="https://cdn-images-1.medium.com/max/800/0*v4pgxvEFE8JvroJv.png" alt="hough1" border="0"></a><br /><br />
</p>


Since,computing &nbsp; ![corner11](https://latex.codecogs.com/gif.latex?E%28u%2Cv%29)  &nbsp; will be computationally challenged, optimisation with Taylor series expansion (only the 1rst order)
is applyed. Some math leads us to: <br/>

![corner12](https://latex.codecogs.com/gif.latex?E%28u%2Cv%29%5Capprox%20%28u%2Cv%29M%5Cbinom%7Bx%7D%7By%7D). &nbsp; 

And finally structure tensor is defined with :
<p align="center">
&nbsp;<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/a617dda21e306dbfbdb7a186b1c203e3f3443867" alt="hough1" border="0"></a><br /><br />
</p>

<li><b>Computation of score R for each found window</b></li>
After fiding windows with large variations, selection of suitable corners is performed. 
It was estimated that the eigenvalues of the matrix can be used to do this. Calculation of a score associated with each such window is given with:  <br/>

![corner2](https://latex.codecogs.com/gif.latex?R%20%3D%20det%28M%29%20-%20k%5Ccdot%20%28trace%28M%29%29%5E%7B2%7D)   <br/>

where &nbsp; ![corner2](https://latex.codecogs.com/gif.latex?det%28M%29%20%3D%20%5Clambda%20_%7B1%7D%20%5Ccdot%20%5Clambda_%7B2%7D) &nbsp; and &nbsp; 
&nbsp; ![corner3](https://latex.codecogs.com/gif.latex?trace%28M%29%20%3D%20%5Clambda%20_%7B1%7D%20&plus;%20%5Clambda_%7B2%7D).  &nbsp;
Here,  &nbsp; ![corner4](https://latex.codecogs.com/gif.latex?%5Clambda%20_%7B1%7D) &nbsp; and  &nbsp; ![corner5](https://latex.codecogs.com/gif.latex?%5Clambda%20_%7B2%7D)  &nbsp; 
are eigenvalues of M, and k is an empirical constant. 


<li><b>Applying a threshold to the score R and important corners selection</b></li>

Depending on the value of R, the window is classified as consisting of flat, edge, or a corner. 
A large value of R indicates a corner, a negative value indicates an edge. 
Also, in order to pick up the optimal values to indicate corners, we find the local maxima as corners within the window which is a 3 by 3 filter.

</ul>

OpenCV implementation of Harris Corner detector can be found in: 

```
Function : cv2.cornerHarris(image,blocksize,ksize,k)
Parameters are as follows :
1. image : the source image in which we wish to find the corners (grayscale)
2. blocksize : size of the neighborhood in which we compare the gradient 
3. ksize : aperture parameter for the Sobel() Operator (used for finding Ix and Iy)
4. k : Harris detector empirical constant parameter (used in the calculation of R)
```
More information can be found at:  <a href="https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html?highlight=cv2.cornerharris#cv2.cornerHarris">OpenCV cv2.cornerHarris documentation</a> 

## Task 4

On the image `images/crossword.jpg`, perform corner detection so that the corners of the crossword are visible. Load the image as **grayscale** and call the cornerHarris function on it. Example call with good initial parameter values:

`corners = cv.cornerHarris(img, 15, 3, 0.05)`

Then set the original image to 255 wherever the corners image is greater than a certain value. Note that the corners image will contain very small values, so a good starting threshold is around 0.0001. Recall the early labs for image thresholding, specifically setting image values where another image is greater than a certain value. Display the resulting image. Try modifying the parameters blockSize, k, or the threshold to better highlight the corners of the crossword.

By combining the Canny edge detector and Harris corner detector, we can segment different parts of images. For example, we can count blood cells in microscopic images, find solutions to Sudoku puzzles, crosswords, or in chess, detect cracks and anomalies in various materials or roads, etc.

# Advanced Feature Detection 

Now we will take a look at more two topics: advanced feature detection extraction algorithms and object detection (using traditional image processing methods). 

Feature extraction is a process of dimensionality reduction by which an initial set of raw data is reduced to more manageable groups for processing. A characteristic of these large data sets is a large number of variables that require a lot of computing resources to process. Feature extraction is the name for methods that select and /or combine variables into features, effectively reducing the amount of data that must be processed, while still accurately and completely describing the original data set.

Object detection is a technique that works to identify and locate objects within an image or video. In this way it provides better understanding and analisys of scenes in images and videos. Specifically, object detection draws bounding boxes around these detected objects, which allow us to locate where said objects are in (or how they move through) a given scene. With this kind of identification and localization, object detection can be used to count objects in a scene and determine and track their precise locations, all while accurately labeling them. 

## Feature Detection and Extraction

What is feature?
A local image feature is a tiny patch in the image that's invariant to image scaling, rotation and change in illumination.
It's like the tip of a tower, or the corner of a window in the image above. Unlike a random point on the background (sky) in the image above,
the tip of the tower can be precise detected in most images of the same scene. It is geometricly (translation, rotation, ...) and photometricly (brightness, exposure, ...) invariant.
A good local feature is like the piece you start with when solving a jigsaw puzzle, except on a much smaller scale.
It's the eye of the cat or the corner of the table, not a piece on a blank wall.
The extracted local features must be:

* Repeatable and precise so they can be extracted from different images showing the same object.
* Distinctive to the image, so images with different structure will not have them.

Due to these requirements, most local feature detectors extract corners and blobs. There is a wealth of algorithms satisfying the above requirements for feature detection (finding interest points on an image) and description
(generating a vector representation for them). They include already learned Harris Corner Detection (in lab 4), and some more advanced algorithms, such as: 

* Scale Invariant Feature Transform (SIFT)
* Speeded-Up Robust Features (SURF)
* Features from Accelerated Segment Test (FAST)
* Binary Robust Independent Elementary Features (BRIEF)
* Oriented FAST and rotated BRIEF (ORB)

The SIFT and SURF algorithms are patented by their respective creators, and while they are free to use in academic and research settings, you should technically be obtaining a license/permission from the creators if you are using them in a commercial (i.e. for-profit) application.

Since is a known fact that ORB performs as well as SIFT on the task of feature detection (while outperforms SURF), in this lab, our focus will be on Oriented FAST and rotated BRIED (ORB).

## Oriented FAST and rotated BRIEF (ORB) 

Oriented FAST and rotated BRIEF (ORB) is a fast robust local feature detector. It is basically a fusion of FAST keypoint detector and BRIEF descriptor
with many modifications to enhance the performance.

ORB is a fusion of FAST keypoint detector and BRIEF descriptor with some added features to improve the performance. FAST is Features from Accelerated Segment Test used to detect features from the provided image. It also uses a pyramid to produce multiscale-features. Now it doesn’t compute the orientation and descriptors for the features, so this is where BRIEF comes in the role.

ORB uses BRIEF descriptors but as the BRIEF performs poorly with rotation. So what ORB does is to rotate the BRIEF according to the orientation of keypoints. Using the orientation of the patch, its rotation matrix is found and rotates the BRIEF to get the rotated version. ORB is an efficient alternative to SIFT or SURF algorithms used for feature extraction, in computation cost, matching performance, and mainly the patents. SIFT and SURF are patented and you are supposed to pay them for its use. But ORB is not patented.


We’ll start by showing the following figure that shows an example of using ORB to match between real world images with viewpoint change. Green lines are valid matches, red circles indicate unmatched points.

ORB  uses an orientation compensation mechanism, making it rotation invariant while learning the optimal sampling pairs.

### Orientation Compensation

ORB uses a simple measure of corner orientation – the intensity centroid [5]. First, the moments of a patch are defined as:

<p align="center">
  <img src="https://gilscvblog.files.wordpress.com/2013/10/figure2.jpg">
</p>

With these moments we can find the centroid, the “center of mass” of the patch as:

<p align="center">
  <img src="https://gilscvblog.files.wordpress.com/2013/10/figure3.jpg?w=300&h=116">
</p>

We can construct a vector from the corner’s center O, to the centroid -OC. The orientation of the patch is then given by:

<p align="center">
  <img src="https://gilscvblog.files.wordpress.com/2013/10/figure4.jpg?w=300&h=53">
</p>

Here is an illustration to help explain the method:

<p align="center">
  <img src="https://gilscvblog.files.wordpress.com/2013/10/angle.jpg">
</p>


Once we’ve calculated the orientation of the patch, we can rotate it to a canonical rotation and then compute the descriptor, thus obtaining some rotation invariance.

We will now see ORB in action through examples and assigments! 

### Assigment 1 - Feature Detection with ORB detector

# Using Convolutions for Feature Detection

## Image Convolution

## Introduction

![convolution](https://i.postimg.cc/x1nPhpHy/conv.png)

Generally, convolution is a mathematical operation between two functions. In the context of this assignment, however, we will focus on discrete 2D convolution between two square images, as that is most relevant for image processing. Convolution is denoted as $I(A) \star k(B)$ where $I(A)$ is an image $I(A) \in \mathbb{R}^{W \times H}$ and $k(B)$ is a matrix $k(B) \in \mathbb{R}^{a \times b}$ indexed by locations $B \in \mathbb{N}^2$ called the **convolutional kernel**. At pixel $(x, y)$, the convolution operation is defined as:

![image](https://github.com/user-attachments/assets/4f1f04a3-45d7-456e-8589-f07ca0bce4e4)

 
Explained differently, the resulting image is produced by sliding the kernel over the input image pixel by pixel. At each pixel location, values where the kernel and the image overlap are multiplied, and all of the products are summed together to form the corresponding pixel's value in the output image.

While mathematically a simple operation, convolution is exceedingly powerful and can produce almost endless transformations of an image. It's most commonly used for filtering --- a convolution can elegantly find patterns in the image and increase their intensity. One such example is the convolution with a kernel called the Prewitt operator:

![image](https://github.com/user-attachments/assets/04907ad3-2d88-4187-a321-10a3926b3e3c)


When convolved with this kernel, the resulting image has high-intensity pixels in regions where vertical edges are present, and low intensity everywhere else. This can be seen in the following image:

![prewitt](https://i.postimg.cc/X7cffKhZ/prewitt-example.png)

Vertical edges necessarily have to have a large jump in values going from left to right or right to left. Otherwise, there would be no perceptible edge. This kernel takes advantage of that fact to accentuate parts of the image where there is such a jump. It does this by replacing each pixel with the difference between the pixels on its left and its right.

This process happens as follows. For each pixel of the input image, the kernel is placed such that it is centered on that pixel. This means that the values of the pixel as well as its neighbors above and below are all multiplied by zero. The neighbors on the left are multiplied by -1, and the ones on the right are multiplied by 1. Summed together, the result represents the sum of the values on the right of the pixel, minus the sum of the values on the left.

To illustrate this, let us consider $1 \times 3$ region of the image where no vertical edges are present:

![image](https://github.com/user-attachments/assets/48876139-8522-4805-a6ab-a0a1963cf46f)


This section of the image does not contain a vertical edge, so the convolution result is a relatively low value. In a standard image with values in $[0, 255)$, 8 would appear almost completely black.

However, consider some section of the image where a vertical edge is indeed present:

![image](https://github.com/user-attachments/assets/6641d826-e60a-46ad-be1d-713ab7cebe92)


The value is now much larger due to the difference between the left and right sides of the image. This example demonstrates how a relatively simple kernel can capture complex features of an image.

Beyond edge detection, there are many commonly used convolution kernels to perform tasks such as blurring, sharpening, or denoising images. A convolutional neural network can leverage the power of the convolution by stringing together sequences of intricate kernels to match complex patterns in the image.

### Resources

- https://en.wikipedia.org/wiki/Kernel_(image_processing)
- https://vincmazet.github.io/bip/filtering/convolution.html
- https://arxiv.org/pdf/1603.07285.pdf


**Finish the code blocks below at the TODO comments. You do not need to change the rest of the code, or code blocks that do not have TODO comments.**

## Understanding kernels

Different convolutional kernels can achieve different effects such as:

 - edge detection
 - image engancement (e.g. sharpening)
 - blurring
 - denoising
 - feature detection

You can see various examples of kernels here:

 - https://en.wikipedia.org/wiki/Kernel_(image_processing)
 - https://setosa.io/ev/image-kernels/

## Using Convolutions for Feature Detection

Assume we want to detect the eyes on the following image:

![](images/woman_darkhair.png)

Notice that the eyes consist of a light region and then a dark region (iris). We may be able to detect the eyes using the following kernel:

![image](https://github.com/user-attachments/assets/a7319d7f-ac2a-4427-87d9-ad818f20827a)


This kernel will take the difference between the left two pixels and the right two pixels. If the pixels on the left are high and on the right are low, the resulting value will be high.

Implement a convolution with this kernel in the next code block.
