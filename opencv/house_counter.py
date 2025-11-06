import cv2
import numpy as np

img = cv2.imread("C:/Users/superpro2005/Desktop/study/python/photo/img_1.png")
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img)

b,g,r = cv2.split(img)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,threshold = cv2.threshold(g,230,255,cv2.THRESH_BINARY_INV)
contures,hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# mask  = np.zeros_like(gray_img)
# cv2.drawContours(mask,contures,-1,255,1)
# result = cv2.bitwise_and(img,img,mask=mask)
min_area = 100
house_count = 0
min_width = 40


for cnt in contures:
    epsilon = 0.002 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(img, [approx], -1, 255, 1)

    mask_obj = np.zeros_like(threshold,dtype="uint8")
    cv2.drawContours(mask_obj,[cnt],-1,255,-1)
    masked_img = cv2.bitwise_and(threshold,threshold, mask =mask_obj)
    original_roi = cv2.bitwise_and(img, img, mask=mask_obj)
    white_pixels = cv2.countNonZero(masked_img)
    mean_color = cv2.mean(original_roi)[0]
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0

    if (cv2.contourArea(cnt) > min_area) and (mean_color < 100) and (0.05 < aspect_ratio < 2.5) :
        rect = cv2.minAreaRect(cnt)
        house_count += 1
        position = tuple(map(int, rect[0]))

        cv2.putText(img, str(house_count), position, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

# img_with_contours = img.copy()
# cv2.drawContours(img_with_contours, contures, -1, (0, 0, 255), 1)

cv2.imshow("gray",img)
cv2.imshow("threshold",threshold)

cv2.waitKey(0)