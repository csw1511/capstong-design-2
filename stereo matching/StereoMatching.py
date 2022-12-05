import numpy as np
import cv2
from matplotlib import pyplot as plt

def stereoMatching(leftImg,rightImg):
    
    #이미지의 크기를 알아냄
    rows = leftImg.shape[0]
    cols = leftImg.shape[1]
    
    # 좌우 DSI 크기만큼 0을 채움
    # 이게 결과물이 됨
    leftDSI=np.zeros((rows,cols))
    rightDSI=np.zeros((rows,cols))

    #대각선이 아닌 방향으로 한 칸 움직일 때의 코스트
    costPerBlock = 21
    
    #커널은 여기서 지정
    #3*3 크기
    #지금은 Gaussian filter로 설정
    kernel = [0.0625, 0.125, 0.0625,
              0.125, 0.25, 0.125,
              0.0625, 0.125, 0.0625]
    
    # 행 단위로 반복
    for c in range (0,rows):
        print("current row :",c)
        # Dytnamic programming에 의해 최적 루트를 저장하기 위한 배열
        dsiMat=np.zeros((cols,cols))
        # 코스트를 저장할 배열, 좌우 이미지의 c행의 각 열을 비교
        colMat=np.zeros((cols,cols))
        
        # 첫째 행, 첫째 열은 무조건 해당 방향으로 한칸씩 이동해야 최적이므로
        # 원점에서 떨어진 거리 * 블록당 코스트를 저장해 둠
        for i in range(0,cols):
            colMat[i][0] = i*costPerBlock
            colMat[0][i] = i*costPerBlock

        
        # 두 이미지의 같은 행의 열을 서로 비교함
        for k in range (0,cols):
            for j in range(0,cols):                        
                #좌우 이미지의 같은 픽셀의 값 차이를 구함
                #후에 이 값이 행이나 열 방향으로 움직이는 코스트보다 작다면, 현재 점이 최소 코스트를 가지는 경로의 일부가 됨
                #즉, 대각선 방향으로 진행하게 됨
                if leftImg[c][k] < rightImg[c][j]:
                    match_cost=rightImg[c][j]-leftImg[c][k]                    
                else:
                    match_cost=leftImg[c][k]-rightImg[c][j]
                
                # 최소 코스트를 찾음
                #대각선으로 이동하는 경우, 이전 코스트에 앞서 저장한 코스트를 더해줌
                min1=colMat[k-1][j-1]+match_cost 
                
                #행이나 열 방향으로 이동하는 경우, 이전 코스트에 블록당 코스트 만큼 더해줌
                min2=colMat[k-1][j]+costPerBlock 
                min3=colMat[k][j-1]+costPerBlock
                
                #앞서 저장한 세 개의 코스트중 가장 작은 값으로 저장
                colMat[k][j]=cmin=min(min1,min2,min3)
                
                # 최적 루트를 기록함
                if min1 == cmin:
                    dsiMat[k][j]=1
                if min2 == cmin:
                    dsiMat[k][j]=2
                if min3 == cmin:
                    dsiMat[k][j]=3
        

        # 끝까지 도달했으면 이제 되짚어가면서 DSI에 값을 저장함
        i=cols-1
        j=cols-1
        
        while i > 0 and  j > 0:
            if dsiMat[i][j] == 1:
                leftDSI[c][i]=np.absolute(i-j)
                rightDSI[c][j]=np.absolute(j-i)
                i=i-1
                j=j-1
            elif dsiMat[i][j] == 2:
                leftDSI[c][i]=0
                i=i-1
            elif dsiMat[i][j] == 3:
                rightDSI[c][j]=0
                j=j-1
    # 완성된 DSI의 중간중간 구멍이 있는 부분은 필터를 통해 bluring 시킴
    blurDisp=np.zeros((rows,cols)) # bluring 시킨 DSI
    for i in range(0,rows):
        for j in range(0,cols):
            filterValues = [0]*9
            index = -1
            for _i in range(i-1, i+2):                             
                for _j in range(j-1, j+2):
                    index +=1 
                    if _i < 0 or _i > rows-1:
                        continue   
                    if _j < 0 or _j > cols-1:
                        continue
                    #커널에 입력할 값 수집
                    filterValues[index] = rightDSI[_i][_j]
            blurDisp[i][j] = filtering(filterValues, kernel)
    
    #dynamic programming만 사용한 것
    plt.subplot(121),plt.imshow(rightDSI, cmap = 'gray')
    plt.title('Disparity'), plt.xticks(), plt.yticks()
    
    #dynamic programming 후 커널을 사용해 보정을 거친 것
    plt.subplot(122),plt.imshow(blurDisp, cmap = 'gray')
    plt.title('blur Disparity'), plt.xticks(), plt.yticks()
    plt.show()
        
def filtering(vals,kernel):
    result = 0
    for i in range(0,len(kernel)):
        result += vals[i] * kernel[i]
    return result
     
def main():
    
    # 이미지 읽기
    leftImg = cv2.imread("1left.png",0)
    leftImg = np.asarray(leftImg, dtype=np.uint8) 
    rightImg = cv2.imread("1right.png",0)
    rightImg = np.asarray(rightImg, dtype=np.uint8)
    
    # Streo matching 수행
    stereoMatching(leftImg,rightImg)
    
main()