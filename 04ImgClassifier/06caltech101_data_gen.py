# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

categories = ['chair', 'camera','butterfly','elephant','flamingo']
nb_classes = len(categories)

image_w = 64
image_h = 64

# npz 파일 불러오기
data = np.load('./saveFiles/caltech_5object.npz')
# np.savez() 함수로 저장시 이미 데이터를 훈련용, 테스트용으로 분리했음
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

# 픽셀 데이터를 0~1사이로 변환하여 신경망 학습 안정화
X_train = X_train.astype('float') / 256
X_test = X_test.astype('float') / 256
# 데이터 형태 출력(샘플 개수, 64, 64, 채널 수)
print('X_train shape:',X_train.shape)

train_datagen = ImageDataGenerator(
    rotation_range=10, # 이미지를 -10~10도 사이로 랜덤 회전
    width_shift_range=0.1, # 가로 방향으로 최대 10% 이동(랜덤)
    height_shift_range=0.1, #세로 방향으로 최대 10% 이동
    horizontal_flip=True, # 이미지를 수평방향으로 반전(뒤집기)
    zoom_range=0.1, # 최대 10% 확대 or 축소(랜덤)
    fill_mode='nearest' # 변환 시 생긴 빈 픽셀을 가장 가까운 픽셀 값으로 채움
)

# 모델 생성 : 레이어를 순차적으로 쌓는 방식으로 Sequential 모델 생성
model = Sequential()

# 입력층 - 합성곱 층
model.add(Conv2D(32, (3,3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu')) # 활성화 함수 ReLU 적용(비선형성 추가)
model.add(MaxPooling2D(pool_size=(2,2))) # 2*2최대 풀링 적용
model.add(Dropout(0.25)) # 과적합 방지를 위한 드롭아웃 적용(25% 뉴런 비활성화)

# 은닉층1 - 합성곱 층
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))

# 은닉층2 - 합성곱 층
model.add(Conv2D(64, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 은닉층3 - 완전 연결층
model.add(Flatten()) # 2D 특징맵을 1D 벡터로 변환
model.add(Dense(512)) # 완전연결층(512개의 뉴런)
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 출력층 : 클래스의 수만큼 뉴런 생성
model.add(Dense(nb_classes))
# 다중 클래스 분류를 위한 소프트맥스 활성화 함수 지정
model.add(Activation('softmax'))

# 모델 컴파일 : 손실 함수 , 옵티마이저, 평가 지표
model.compile(loss='binary_crossentropy', # 손실함수 지정 
              optimizer='rmsprop', # RMSprop 옵티마이저 사용 
              metrics=['accuracy']) # 평가 지표로 정확도 사용

# 데이터 증강을 적용한 모델 훈련
'''
step_per_epoch : 한 epoch(전체 학습 데이터 1회 학습) 당 몇번의 batch를
    수행할지 지정함.
'''
model.fit(train_datagen.flow(X_train, Y_train, batch_size=32), epochs=50, steps_per_epoch=len(X_train)//32)

score = model.evaluate(X_test, Y_test)
print('loss =',score[0])
print('accuracy =',score[1])
