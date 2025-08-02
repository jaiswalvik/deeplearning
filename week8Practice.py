import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import io

#Read and display the image

X = io.imread('https://caltech-prod.s3.amazonaws.com/main/images/feynman01-NEWS-WEB.width-600_tSwRQP5.jpg')

X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(6,4))
plt.imshow(X,cmap='gray')
plt.show()


#Define kernels vary the size by changing m and n 
m=12
n=12
K = (1/(m*n))*np.ones((m,n))
O = cv2.filter2D(X,ddepth=-1,kernel=K)
plt.imshow(O,cmap='gray')
plt.show()
