[toc]
# Tips

지금까지의 개념들을 알아본다.

**Maximum Likelihood Estimation**

**Optimization via Gradient Descent**

**Overfitting and Regrularization**

**Training and Test Dataset**

**Learning Rate**

**Data Preprocessing**

## MLE(Maximum Likelihood Estimation)

<img src="C:\Users\이현동\AppData\Roaming\Typora\typora-user-images\image-20200404221620517.png" alt="image-20200404221620517" style="zoom:80%;" />

MLE(최대가능도 방법)

압정이 떨어질 결과 예측

- Class 1
- Class 2

이 확률 분포를 알고 싶다.

예측해야 하는 값(Class1, Class2)이 오로지 두 개이다.

-> Bernoulli Distribution으로, 이항 분포(베르누이 분포)의 형태로 나타난다.



0과 1의 binary classfication을 수행하자!

### 실험

- 반복 - 100번
  - class 1 k= 27 (observation 관측값)

<img src="C:\Users\이현동\AppData\Roaming\Typora\typora-user-images\image-20200404230353561.png" alt="image-20200404230353561" style="zoom:80%;" />

<img src="C:\Users\이현동\AppData\Roaming\Typora\typora-user-images\image-20200404230658985.png" alt="image-20200404230658985" style="zoom:80%;" />

- Likelihood (가능도)  관측값(표본)에 대한 확률 부여

## **Optimization via Gradient Descent**

- 확률 분포 함수의 파라미터(theta)를 찾는 과정
  - 기울기를 활용하기 -> gradient ascent

$\theta := \theta - \alpha \delta L(x_i\theta)$ ($\alpha$는 lr(learning rate))

## Overfitting

MLE같은 경우는 숙명적으로  overfitting이 따르게 된다.

- O/X 혹은 0/1을 결정하는 binary

<img src="C:\Users\이현동\AppData\Roaming\Typora\typora-user-images\image-20200404231732943.png" alt="image-20200404231732943" style="zoom:80%;" />

dicision boundary 경계선이 아닌 주어진 데이터에 대해서 과도하게 fitting 되어버린 케이스, 상황을 **overfitting**이라고 한다.



데이터를 가장 잘 설명하는 확률 분포 함수를 찾다 보니 overfitting이 일어날 수밖에 없다.

### observation 구성

- Training Set 0.8
- Development Set(Validation Set) 0~0.1
- Test Set
  - Training Set과 비슷할 것이라는 믿음

Dev Set을 통해 Test Set에 대한 Overfitting 역시 검증하는 과정을 거친다.

<img src="C:\Users\이현동\AppData\Roaming\Typora\typora-user-images\image-20200404232331957.png" alt="image-20200404232331957" style="zoom:80%;" />

- epoch 선택



## Overfitting - Regularization

- More Data
- Less features
  - 적을 수록 overfitting을 막을 수 있다..?

- Regularization
  - Early Stopping : Validation Loss가 더이상 작아지지 않을 때.
  - Reducing Network Size
  - Weight Decay
  - Droptout
  - Batch Normalization

### DNN

<img src="C:\Users\이현동\AppData\Roaming\Typora\typora-user-images\image-20200404233014028.png" alt="image-20200404233014028" style="zoom:80%;" />

- overfitting이 될 때까지
  - layer의 깊이와 넓이를 늘려줌
- 확인 후(validation set loss가 높아지고, training set도 낮아지고 있을 떄)
  - regularization을 통해 term 추가
  - 이를 반복하면 좋은 성능의 모델을 만들 수 있다.

# 실습

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```

## Training and Test Dataset

```python
x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7],
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 1, 0, 0])

x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])
```

## Model

```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init()
        self.linear = nn.Linear(3, 3)
        
    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()

#optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

## Training

```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs): # + 1...????
        
        #H(x) 계산
        prediction = model(x_train)  # SoftmaxClassfierModel()
        
        # cost 계산
        cost = F.cross_entropy(prediction, y_train)
        
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

## Test (Validation)

```python
def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct
```

