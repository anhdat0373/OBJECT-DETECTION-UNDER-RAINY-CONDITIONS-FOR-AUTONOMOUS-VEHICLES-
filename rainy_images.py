import cv2
import numpy as np

img = cv2.imread('mua2.jpg')

# Làm mịn giữ cạnh
dst = cv2.bilateralFilter(img, 9, 75, 75)

# Tăng tương phản
lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
l,a,b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

cv2.imwrite('derain2.jpg', final)