from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import numpy as np

root_dir = './download'
categories = ['Gyudon', 'Ramen','Sushi','Okonomiyaki','Karaage']

nb_classes = len(categories)

image_size = 50

X = []
Y = []

for idx, cat in enumerate(categories):
    # 경로를 조립한 후 jpg 파일 가져오기
    image_dir = root_dir+'/'+cat
    # 각 카테고리 폴더에 있는 모든 jpg 파일을 리스트로 반환
    files = glob.glob(image_dir+'/*.jpg')
    
    # 파일 갯수만큼 반복
    for i, f in enumerate(files):
        # 이미지 파일 오픈 및 변환, 사이즈 조절
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        # 이미지 픽셀값을 넘파이 배열로 변환
        data = np.asarray(img)

        # 넘파이 배열로 변환된 이미지와 카테고리 인덱스 추가
        X.append(data)
        Y.append(idx)

# 리스트를 NumPy 배열로 변환
X = np.array(X)
Y = np.array(Y)

# 학습용 데이터와 테스트용 데이터 분류
'''
X_train : 학습용(훈련용) 입력 데이터
X_test : 테스트용 입력 데이터
Y_train : 학습용(훈련용) 정답 레이블
Y_test : 테스트용 정답 레이블
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

np.savez('./saveFiles/japanese_food.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
print('Task Finished..!!', len(Y))