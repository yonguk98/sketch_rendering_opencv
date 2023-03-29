import cv2
import numpy as np
kernel_table = [
    {'name': 'Sharpen (5)',     'kernel': np.array([[ 0, -1,  0],
                                                    [-1,  5, -1],
                                                    [ 0, -1,  0]])},
    {'name': 'Sobel Y',         'kernel': np.array([[-1, -2, -1],
                                                    [ 0,  0,  0],
                                                    [ 1,  2,  1]])},
    {'name': 'Gradient X',      'kernel': np.array([[-1,  1]])},
    {'name': 'Gradient Y',      'kernel': np.array([[-1], [1]])},
    {'name': 'Laplacian (4)',   'kernel': np.array([[ 0, -1,  0], # Alternative: Laplacian
                                                    [-1,  4, -1],
                                                    [ 0, -1,  0]])},]
# 이미지를 불러옵니다.
img1 = cv2.imread('hw2/image1.jpg') 
img2 = cv2.imread('hw2/image2.jpg') 
img3 = cv2.imread('hw2/image3.jpg')

img4 = cv2.imread('hw2/image2.jpg', cv2.IMREAD_GRAYSCALE)

threshold1 = 250    
threshold2 = 1000
aperture_size = 7

# 이미지를 회색조로 변경합니다.
gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) 

# 가우시안 블러(흐리게) 처리합니다.
gray = cv2.medianBlur(gray, 5) 

sharpen, sh_kernel = kernel_table[0].values()
laplacian, la_kernel = kernel_table[4].values()

# 엣지 검출을 수행합니다.
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9) # 컬러 이미지로 변경합니다.
color = cv2.bilateralFilter(img3, 9, 400, 400) 

sh_result = cv2.filter2D(img3, cv2.CV_64F, sh_kernel)        # Note) dtype: np.float64
sh_result = cv2.convertScaleAbs(sh_result)

la_result = cv2.filter2D(gray, cv2.CV_64F, la_kernel)        # Note) dtype: np.float64
la_result = cv2.convertScaleAbs(la_result)

edge  = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)

la_result_3ch = np.repeat(la_result[:, :, np.newaxis], 3, -1)

# 컬러 이미지와 엣지를 합성합니다.
cartoon = cv2.bitwise_and(color, color, mask=edge)

# 결과를 출력합니다.
cv2.imshow("Cartoon", cartoon) 
cv2.waitKey(0) 
cv2.destroyAllWindows()