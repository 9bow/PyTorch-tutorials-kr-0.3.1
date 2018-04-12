# -*- coding: utf-8 -*-
"""
PyTorch: Variable과 autograd
-----------------------------

하나의 은닉 계층(Hidden Layer)과 편향(Bias)이 없는 완전히 연결된 ReLU 신경망에
유클리드 거리(Euclidean Distance)의 제곱을 최소화하여 x로부터 y를 예측하도록
학습하겠습니다.

PyTorch Variable 연산을 사용하여 순전파를 계산하고, PyTorch autograd를 사용하여
변화도(Gradient)를 계산하는 것을 구현하겠습니다.

PyTorch Variable은 PyTorch Tensor의 래퍼(Wrapper)이며, 연산 그래프(Computational
Graph)에서 노드(Node)로 표현(represent)됩니다. x가 Variable일 때, x.data는 그 값을
갖는 Tensor이며 x.grad는 어떤 스칼라 값에 대해 x의 변화도를 갖는 또 다른 Variable
입니다.

PyTorch Variable은 PyTorch Tensor와 동일한 API를 제공합니다: Tensor에서 할 수 있는
(거의) 모든 연산은 Variable에서도 할 수 있습니다; 차이점은 autograd가 자동으로
변화도를 계산할 수 있다는 것입니다.
"""
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # GPU에서 실행하려면 이 주석을 제거하세요.

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉 계층의 차원이며, D_out은 출력 차원입니다:
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성하고, Variable로
# 감쌉니다. requires_grade=False로 설정하여 역전파 중에 이 Variable들에 대한
# 변화도를 계산할 필요가 없음을 나타냅니다.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성하고, Variable로
# 감쌉니다. requires_grad=True로 설정하여 역전파 중에 이 Variable들에 대한
# 변화도를 계산할 필요가 있음을 나타냅니다.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 순전파 단계: Variable 연산을 사용하여 y 값을 예측합니다. 이는 Tensor를 사용한
    # 순전파 단계와 완전히 동일하지만, 역전파 단계를 별도로 구현하지 않기 위해 중간
    # 값들(Intermediate Value)에 대한 참조(Reference)를 갖고 있을 필요가 없습니다.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Variable 연산을 사용하여 손실을 계산하고 출력합니다.
    # loss는 (1,) 모양을 갖는 Variable이며, loss.data는 (1,) 모양의 Tensor입니다;
    # loss.data[0]은 손실(loss)의 스칼라 값입니다.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # autograde를 사용하여 역전파 단계를 계산합니다. 이는 requires_grad=True를
    # 갖는 모든 Variable에 대한 손실의 변화도를 계산합니다. 이후 w1.grad와 w2.grad는
    # w1과 w2 각각에 대한 손실의 변화도를 갖는 Variable이 됩니다.
    loss.backward()

    # 경사하강법(Gradient Descent)을 사용하여 가중치를 갱신합니다; w1.data와
    # w2.data는 Tensor이며, w1.grad와 w2.grad는 Variable이고, w1.grad.data와
    # w2.grad.data는 Tensor입니다.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
    w1.grad.data.zero_()
    w2.grad.data.zero_()
