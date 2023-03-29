import cv2
import numpy as np

# 이미지를 불러옵니다.
img = cv2.imread('image1.jpg') 

# 이미지를 회색조로 변경합니다.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# 가우시안 블러(흐리게) 처리합니다.
blur = cv2.medianBlur(gray, 9) 

# 엣지 검출을 수행합니다.
edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9) # 컬러 이미지로 변경합니다.
color = cv2.bilateralFilter(img, 9, 300, 300) 

# 컬러 이미지와 엣지를 합성합니다.
cartoon = cv2.bitwise_and(color,color,mask=edges)

# 엣지 이외의 평탄한 부분은 흰색으로 처리
sketch = cv2.divide(gray,blur,scale=255)

# 사진 합성을 위해 채널바꾸기
sketch = np.repeat(sketch[:, :, np.newaxis], 3, -1)

merge = np.hstack((cartoon,sketch))

# 결과를 출력합니다.
cv2.imshow("Cartoon", merge) 
cv2.waitKey(0) 
cv2.destroyAllWindows()