#Import the libraries
from skimage.metrics import structural_similarity as compare_ssim, mean_squared_error, peak_signal_noise_ratio
import cv2
import imutils

# load the two input images
imageA = cv2.imread('C:\\Users\\hegde\\OneDrive\\Desktop\\imgCompareModel\\input1.png')
imageB = cv2.imread('C:\\Users\\hegde\\OneDrive\\Desktop\\imgCompareModel\\input2.png')

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two images
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# calculate Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
mse = mean_squared_error(grayA, grayB)
psnr = peak_signal_noise_ratio(grayA, grayB)
print("MSE: {}".format(mse))
print("PSNR: {}".format(psnr))

# threshold the difference image
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and draw it on both input images
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# display the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)


##########################################################
#THE METRICS LISTED:
#1)SSIM (Structural Similarity Index): SSIM is a metric that quantifies the similarity between two images.It takes into account the luminance, contrast, and structure of the images. The SSIM value ranges from -1 to 1, where a value of 1 indicates that the images are identical.

#2)MSE (Mean Squared Error): MSE is a measure of the average squared difference between the pixels of the two images. It provides a quantitative measure of the overall difference between the images, where lower values indicate greater similarity.

#3)PSNR (Peak Signal-to-Noise Ratio): PSNR is a metric that measures the quality of a reconstructed image compared to the original image. It is expressed in decibels (dB) and higher values indicate better image quality.