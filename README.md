# GMM_from_Scratch
The MNIST handwritten digit dataset  is downloaded and unzipped (training set images and training set labels) from http://yann.lecun.com/exdb/mnist/. 
It contains 28×28-pixel images for the hand-written digits {0,1,...,9} by storing each pixel value ranging between 0 and 255, and their corresponding true labels.
Load the data into Python. Preprocess the data by compressing each image to 1/4 of the original size in the following way: Divide each 28×28 image into 2×2 non-overlapping blocks. 
Calculate the mean pixel value of each 2×2 block, and create a new 14×14 image. This preprocessing step will drastically help computation. We will be clustering the digits {0, 1, 2, 3, 4} in this homework. For visualization purpose, we view each data sample as 14×14 matrix. 
For using in an algorithm, treat each sample as a vector - you just simply stack each column of the 14 × 14 matrix into a 196 dimensional vector.

We then implemented the EM algorithm to cluster the MNIST data set.
(i) Program the EM algorithm you derived for mixture of spherical Gaussians. Assume 5 clusters. Terminate the algorithm when the fractional change of the log-likelihood goes under 0.0001.

(ii) Program the EM algorithm you derived for mixture of diagonal Gaussians. Assume 5 clusters. Terminate the algorithm when the fractional change in the log-likelihood goes under 0.0001. (Try 3 random initializations and present the best one in terms of maximizing the likelihood function).
Note that to assign a sample xi to a cluster j, you first calculate Fij using the parameters from the last iteration of EM algorithm you implemented. Next, assign sample xi to the cluster j for which Fij is maximum, i.e., the probability of sample i belonging to cluster j is maximum. Recall that the dataset has the true labels for each classes. Calculate the error of the algorithm (for the two different model). Here, error is just the number of mis-clustered samples divided by the total number of samples. 

Hint: For these implementations, we ran into three different problems. We apply these following tricks to solve each problem.
• 1) Use the log-sum-exp trick to avoid underflow on the computer. We ran into this problem when computing the log-likelihood. That is, when wecalculate log 􏰀j expaj for some sequence of variables aj , calculate instead A + log 􏰀j expaj −A where A = maxj aj.

• 2) Some pixels in the images do not change throughout the entire dataset. (For example, the top-left pixel of each image is always 0, pure white.) To solve this, after updating the covariance matrix Σj for the mixture of diagonal Gaussians, add 0.05Id to Σj (ie: add 0.05 to all the diagonal elements).

• 3) Be mindful of how you initialize Σj. Note that for a diagonal matrix Σj, the deter- minant |Σj| is the product of all the diagonal elements. Setting each diagonal element to a number too big at initialization will result in overflow on the computer.
