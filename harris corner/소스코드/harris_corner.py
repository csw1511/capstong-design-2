import numpy as np
import cv2 as cv

# 이미지파일을 불러와서 ima_@에 read한다.
file_1 = 'derivative_image.jpg'
img_1 = cv.imread(file_1)
file_2 = 'gaussian_filter_image.jpg'
img_2 = cv.imread(file_2)
file_3 = 'laplacian_filter_image.jpg'
img_3 = cv.imread(file_3)

# 해리스 코너를 검출하기 위한 코드.
gray1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
dst1 = cv.cornerHarris(gray1, 4, 3, 0.04)

gray2 = cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
dst2 = cv.cornerHarris(gray2, 4, 3, 0.04)

gray3 = cv.cvtColor(img_3,cv.COLOR_BGR2GRAY)
gray3 = np.float32(gray3)
dst3 = cv.cornerHarris(gray3, 4, 3, 0.04)
# cornerHarris(a, b, c, d),
# a = float32타입의 그레이스케일 이미지. 위의 2줄의 코드는 각각 이미지를 그레이스케일,float32타입으로 변환해주는 코드이다.
# b = 코너 검출을 위해 고려할 이웃 픽셀의 범위
# c = sobel 미분에 사용된 인자값
# d = 경험적 상수로 0.04~0.06의 값을 넣는데 이 코드에선 0.04를 넣었음. 유도된 R 식에서는 k로 표현

#검출된 코너 부분을 확대하기 위해 dilate를 적용
dst1 = cv.dilate(dst1,None)
dst2 = cv.dilate(dst2,None)
dst3 = cv.dilate(dst3,None)

# 원본 이미지에서 적당한 부분을 붉은색으로 표시. max()앞에 곱한 상수(이 코드에선 0.01)를
# 조절하면 검출된 코너를 최적화하여 화면에 표시 가능
img_1[dst1>0.01*dst1.max()]=[0,0,255]
img_2[dst2>0.01*dst2.max()]=[0,0,255]
img_3[dst3>0.01*dst3.max()]=[0,0,255]

# 이미지를 출력.
cv.imshow('dst1',img_1)
cv.imshow('dst2',img_2)
cv.imshow('dst3',img_3)


cv.imwrite('corner_detection_derivative_image.jpg', img_1)
cv.imwrite('corner_detection_gaussian_filter_image.jpg', img_2)
cv.imwrite('corner_detection_laplacian_filter_image.jpg', img_3)


