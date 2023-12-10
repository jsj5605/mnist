# mnist 이미지 분류
- **[MNIST](https://ko.wikipedia.org/wiki/MNIST_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4) (Modified National Institute of Standards and Technology) database**
- 흑백 손글씨 숫자 0-9까지 10개의 범주로 구분해놓은 데이터셋
- 하나의 이미지는 28 * 28 pixel 의 크기
- 6만개의 Train 이미지와 1만개의 Test 이미지로 구성됨.

## 1. import 
``` python
import torch
import torch.nn as nn  # 딥러닝 모델을 구성하는 함수들의 모듈.
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import os
```
## 2. device 설정
``` python
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
```

## 3. 하이퍼파라미터 및 변수 설정
``` python
BATCH_SIZE = 256 # 모델의 파라미터를 업데이트할때 사용할 데이터의 개수. 한번에 몇개 데이터를 입력할 지.
N_EPOCH = 20   # 전체 train dataset을 한번 학습한 것을 1 epoch
LR = 0.001     # 학습률. 파라미터 update할 때 gradient값에 곱해줄값. 
#                (gradient를 새로운 파라미터 계산할 때 얼마나 반영할지 비율)

DATASET_SAVE_PATH = "datasets"  # 데이터셋을 저장할 디렉토리 경로.
MODEL_SAVE_PATH = "models"  # 학습-평가가 끝난 모델을 저장할 디렉토리 경로.

os.makedirs(DATASET_SAVE_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
```

## 4. mnist dataset loading
### data set 준비
``` python
train_set = datasets.MNIST(root=DATASET_SAVE_PATH,  # 데이터셋을 저장할 디렉토리 경로
                         train=True, # trainset(훈련용): True, testset(검증용): False
                         download=True, # root 에 저장된 데이터파일들이 없을때 다운로드 받을지 여부
                         transform=transforms.ToTensor() # 데이터 전처리.
                          )
# ToTensor(): ndarray, PIL.Image객체 를 torch.Tensor로 변환. 
#             Pixcel값 정규화(normalize): 0 ~ 1 실수로 변환

test_set = datasets.MNIST(root=DATASET_SAVE_PATH, 
                         train=False, 
                         download=True,
                         transform=transforms.ToTensor())
```

### dataloader 준비
``` python
# Dataset을 모델에 어떻게 제공할지를 설정. => 학습/평가시 설정된대로 데이터를 loading
## 훈련용 DataLoader
train_loader = DataLoader(train_set, # Dataset
                          batch_size=BATCH_SIZE, # batch_size를 설정.
                          shuffle=True, # 한 epoch이 끝나면 다음 Epoch 전에 데이터를 섞을지 여부.
                          drop_last=True, # 마지막 batch의 데이터수가 batch_size보다 적을 경우 버릴지(학습에 사용안함) 여부
                         )
## 평가용 DataLoader
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
```

## 5. 모델 정의
```python
# class로 정의:  nn.Module 을 상속해서 정의
class MnistModel(nn.Module):
    
    def __init__(self):
        """
        모델 객체 생성시 모델을 구현(정의)할 때 필요한 것들을 초기화.
        필요한 것: Layer들. 등등
        """
        super().__init__()
        
        #### 784(pixcel수) -> 128 개로 축소
        self.lr1 = nn.Linear(784, 128)  # input feature크기, output size
        #### 128 feature -> 64 축소
        self.lr2 = nn.Linear(128, 64)
        ### 64 feature -> 출력결과 10 (각 범주의 확률)
        self.lr3 = nn.Linear(64, 10)
        #### Activation(활성) 함수 -> 비선형함수 : ReLU
        self.relu = nn.ReLU() # f(x) = max(x, 0)
        
    def forward(self, x):
        """
        input data를 입력 받아서 output 데이터를 만들때 까지의 계산 흐름을 정의
        ===> forward propagation
        parameter
            x : 입력데이터
        return
            torch.Tensor: 출력데이터(모델 예측결과.)
        """
        # init에서 생성한 함수들을 이용해서 계산
        ###  x -> 1차원으로 변환-> lr1 -> relu -> lr2 -> relu -> lr3 -> output
        # input (batch_size, channel, height, width) => (batch_size, 전체pixcel)
        x = torch.flatten(x, start_dim=1) # (b, c, h, w)->(b, c*h*w)
        
        x = self.lr1(x) # 선형공식(함수)
        x = self.relu(x)# 비선형함수(relu)
        
        x = self.lr2(x)
        x = self.relu(x)
        
        output = self.lr3(x)
        return output
```
```python
# 모델 생성
model = MnistModel()
print(model)

output:
MnistModel(
  (lr1): Linear(in_features=784, out_features=128, bias=True)
  (lr2): Linear(in_features=128, out_features=64, bias=True)
  (lr3): Linear(in_features=64, out_features=10, bias=True)
  (relu): ReLU()
)
```

## 6. train
### 모델, loss function, optimizer 설정
```python
model = model.to(device) # model을 device로 옮긴다

##### loss function
# 다중분류문제: crossentropy, 이진분류문제: binary crossentropy ==> log loss
# 다중분류: label이 여러개, 이진분류: yes/no 분류
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), # 최적화 대상 파라미터들
                             lr=LR)  #학습률
```
### 학습 및 검증
```python
import time # 학습 시간 체크

## 학습 => train(훈련) + validation(1 epoch 학습한 모델성능 검증)
# 에폭(epoch)별 학습결과를 저장할 리스트들
train_loss_list = [] # train set으로 검증했을 때 loss (loss_fn계산값)
val_loss_list = []   # test set으로 검증했을 때 loss
val_accuracy_list = [] # test set으로 검증했을 때 accuracy(정확도)-전체중 맞은 비율

start = time.time()
# Train
for epoch in range(N_EPOCH):
    #################################
    # Train
    #################################
    model.train() # 모델을 train모드로 변경.
    train_loss = 0.0 # 현재 epoch의 학습 결과 loss를 저장할 변수.
    # 배치단위로 학습
    for X_train, y_train in train_loader:  # batch 단위 (input, output) 튜플로 반환.
        # 1. X, y를 device로 옮긴다. (model, X, y는 같은 device상에 위치해야한다.)
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        # 2. 추론
        pred = model(X_train)
        
        # 3. Loss 계산
        loss = loss_fn(pred, y_train) # args 순서: (모델예측값,  정답)
        
        # 4. 모델의 파라미터 업데이트(최적화)
        ## 1. 파라미터의 gradient값들을 초기화
        optimizer.zero_grad()
        ## 2. gradient 계산 ===> 계산결과는 파라미터.grad 속성에 저장.
        loss.backward()
        ## 3. 파라미터(weight, bias) 업데이트 ( 파라미터 - 학습률*grad)
        optimizer.step()
        
        #### 현재 batch의 loss값을 train_loss변수에 누적
        train_loss += loss.item()  # Tensor -> 파이썬 값
     
    # 1 epoch학습 종료 
    # epoch의 평균 loss를 계산해서 리스트에 저장. (train_loss: step별 loss를 누적)
    train_loss_list.append(train_loss / len(train_loader))  #step수 나눔.
    
    ########################################
    # validate(검증) - test(validation) set(학습할 때 사용하지 않았던 데이터셋)
    ########################################
    model.eval() # 모델을 검증(평가) 모드로 전환. 
    ## 현재 epoch대한 검증결과(loss, accuracy)를 저장할 변수
    val_loss = 0.0
    val_acc = 0.0
    ### 모델 추정을 위한 연산 - forward propagation
    #### 검증/평가/서비스 -> gradient계산이 필요없다. => 도함수를 계산할 필요 없다.
    with torch.no_grad():
        ## batch 단위로 검증
        for X_val, y_val in test_loader:
            # 1. device로 옮기기
            X_val, y_val = X_val.to(device), y_val.to(device)
            # 2. 모델을 이용해 추론
            pred_val = model(X_val)
            # 3. 검증 
            ## 1. loss 계산 + val_loss에 누적
            val_loss = val_loss + loss_fn(pred_val, y_val).item()
            ## 2. 정확도(accuarcy): 맞은것개수/전체개수
            val_acc = val_acc + torch.sum(pred_val.argmax(axis=-1) == y_val).item()
        # test set 전체에대한 검증이 완료 => 현 epoch 에 대한 검증 완료
        ## val_loss, val_acc 값을 리스트에 저장.
        val_loss_list.append(val_loss / len(test_loader))  # loss 는 step 수 나눔.
        val_accuracy_list.append(val_acc / len(test_loader.dataset) ) # 전체 데이터 개수로 나눈다.
    ## 현재 epoch train 결과를 출력
    print(f"[{epoch+1:2d}/{N_EPOCH:2d}] Train Loss: {train_loss_list[-1]},\
 Val Loss: {val_loss_list[-1]}, Val Accuracy: {val_accuracy_list[-1]}")
    
    
end = time.time()
print(f"학습에 걸린시간: {end-start}초")
```
### 학습로그 시각화
```python
# epoch별 loss, accuracy의 변화흐름을 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(N_EPOCH), train_loss_list, label="train")
plt.plot(range(N_EPOCH), val_loss_list, label="Validation")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(N_EPOCH), val_accuracy_list)
plt.title("Validation accuracy")


plt.tight_layout()
plt.show()
```
<img width="981" alt="image" src="https://github.com/jsj5605/mnist/assets/141815934/1530bc45-a08a-4596-a7d8-957726ccbbd7">

## 7. 학습된 모델 저장 및 불러오기
``` python
save_path = os.path.join(MODEL_SAVE_PATH, "mnist")
os.makedirs(save_path, exist_ok=True)
save_file_path = os.path.join(save_path, "mnist_mlp.pth")
print(save_file_path)

torch.save(model, save_file_path)

### 모델 불러오기
load_model = torch.load(save_file_path)
load_model
```
## 8. 모델 평가
```python
# device로 옮기기
load_model =  load_model.to(device)
# 평가모드 변환
load_model.eval()

test_loss, test_acc = 0.0, 0.0
with torch.no_grad():
    for X, y in test_loader:
        # device 옮기기
        X, y = X.to(device), y.to(device)
        # 추정
        pred = load_model(X)
        # 평가 - loss, accuracy
        test_loss += loss_fn(pred, y).item()
        test_acc += torch.sum(pred.argmax(axis=-1) == y).item()
    test_loss /= len(test_loader) # step수로 나누기
    test_acc /= len(test_loader.dataset)  # 총 데이터수로 나누기
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

output:
Test loss: 0.08874807676620548, Test accuracy: 0.9778
```












                            
