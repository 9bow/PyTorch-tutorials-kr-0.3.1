예제로 배우는 PyTorch
******************************
**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_
  **번역**: `박정환 <http://github.com/9bow>`_

이 튜토리얼에서는 `PyTorch <https://github.com/pytorch/pytorch>`__ 의 기본적인
개념을 포함된 예제를 통해 소개합니다.

PyTorch의 핵심에는 2가지 주요한 특징이 있습니다:

- NumPy와 유사하지만 GPU 상에서 실행 가능한 N차원 Tensor
- 신경망을 구성하고 학습하는 과정에서의 자동 미분

완전히 연결된 ReLU 신경망을 예제로 사용할 것입니다. 이 신경망은 하나의 은닉
계층(Hidden Layer)을 갖고 있으며, 신경망의 출력과 정답 사이의 유클리드 거리
(Euclidean Distance)를 최소화하는 식으로 경사하강법(Gradient Descent)을 사용하여
무작위의 데이터를 맞추도록 학습할 것입니다.

.. Note::
    각각의 에제들은 :ref:`이 페이지의 마지막 부분 <examples-download>` 에서
    살펴볼 수 있습니다.

.. contents:: 목차
	:local:

Tensor
=======

준비 운동: NumPy
-----------------

PyTorch를 소개하기 전에, 먼저 NumPy를 사용하여 신경망을 구성해보겠습니다.

NumPy는 N차원 배열 객체(Object)를 제공하며, 이러한 배열을 조작하기 위한 다양한
함수(function)들을 제공합니다. NumPy는 과학적 분야의 연산을 위한 포괄적인 프레임워크
(Framework)입니다; 이는 연산 그래프(Computational Graph)나 딥러닝, 변화도(Gradient)에
대해서는 알지 못합니다. 하지만 NumPy 연산을 사용하여 순전파 단계와 역전파 단계를
직접 구현함으로써, 2-계층을 갖는 신경망이 무작위의 데이터를 맞추도록 할 수
있습니다:

.. includenodoc:: /beginner/examples_tensor/two_layer_net_numpy.py


PyTorch: Tensor
----------------

NumPy는 훌륭한 프레임워크지만, GPU를 사용하여 수치 연산을 가속화할 수는 없습니다.
현대의 심층 신경망에서 GPU는 종종 `50배 또는 그 이상 <https://github.com/jcjohnson/cnn-benchmarks>`__
의 속도 향상을 제공하기 때문에, 안타깝게도 NumPy는 현대의 딥러닝에는 충분치 않습니다.

이번에는 PyTorch의 기본적인 개념인 **Tensor** 에 대해서 소개하겠습니다.
PyTorch Tensor는 기본적으로 NumPy 배열과 동일합니다: Tensor는 N차원 배열이며,
PyTorch는 Tensor 연산을 위한 다양한 함수들을 제공합니다. NumPy 배열과 같이,
PyTorch Tensor는 딥러닝이나 연산 그래프, 변화도는 알지 못하며, 과학적 분야의 연산을
위한 포괄적인 도구입니다.

그러나 NumPy와는 달리, PyTorch Tensor는 GPU를 활용하여 수치 연산을 가속화할 수
있습니다. GPU에서 PyTorch Tensor를 실행하기 위해서는 단지 새로운 자료형으로
변환(Cast)해주기만 하면 됩니다.

여기에서는 PyTorch Tensor를 사용하여 2-계층의 신경망이 무작위 데이터를 맞추도록
할 것입니다. 위의 NumPy 예제에서와 같이 신경망의 순전파 단계와 역전파 단계는 직접
구현하겠습니다.

.. includenodoc:: /beginner/examples_tensor/two_layer_net_tensor.py


Autograd
========

PyTorch: Variables과 autograd
-------------------------------

위의 예제에서 우리는 신경망의 순전파 단계와 역전파 단계를 수동으로 구현하였습니다.
작은 2-계층 신경망에서 역전파 단계를 직접 구현하는 것은 큰 일이 아니지만,
대규모의 복잡한 신경망에서는 매우 아슬아슬한 일일 것입니다.

다행히도, `자동 미분 <https://en.wikipedia.org/wiki/Automatic_differentiation>`__ 을
사용하여 신경망에서 역전파 단계의 연산을 자동화할 수 있습니다. PyTorch의 **autograd**
패키지는 이 기능을 정확히 제공합니다.
Autograd를 사용할 때, 신경망의 순전파 단계는 **연산 그래프** 를 정의합니다;
그래프의 노드(Node)는 Tensor이며, 엣지(Edge)는 입력 Tensor로부터 출력 Tensor를
만들어내는 함수입니다. 이 그래프를 통해 역전파를 하게 되면 변화도를 쉽게 계산할
수 있습니다.

이는 복잡해보이지만 실제로 사용하는 것은 간단합니다. PyTorch Tensor를 **Variable**
객체로 감싸게 되면, 이 Variable이 연산 그래프에서 노드로 표현(represent)됩니다.
``x`` 가 Variable일 때, ``x.data`` 는 그 값을 갖는 Tensor이며 ``x.grad`` 는 어떤
스칼라 값에 대해 ``x`` 에 대한 변화도를 갖는 또 다른 Variable 입니다.

PyTorch Variable은 PyTorch Tensor와 동일한 API를 제공합니다: Tensor에서 할 수 있는
(거의) 모든 연산은 Variable에서도 할 수 있습니다; 차이점은 연산 그래프를 정의할 때
Variable을 사용하면, 자동으로 변화도를 계산할 수 있다는 것입니다.

여기에서는 PyTorch Variable과 autograd를 사용하여 2-계층 신경망을 구현합니다; 이제
더 이상 신경망의 역전파 단계를 직접 구현할 필요가 없습니다:

.. includenodoc:: /beginner/examples_autograd/two_layer_net_autograd.py

PyTorch: 새 autograd 함수 정의하기
-----------------------------------

Under the hood, autograd의 기본(primitive) 연산자는 실제로 Tensor를 조작하는 2개의
함수입니다. **forward** 함수는 입력 Tensor로부터 출력 Tensor를 계산합니다.
**backward** 함수는 출력 Tensor의 변화도를 받고 입력 Tensor의 변화도를 계산합니다.

PyTorch에서 ``torch.autograd.Function`` 의 하위 클래스(subclass)를 정의하고
``forward`` 와 ``backward`` 함수를 구현함으로써 쉽게 사용자 정의 autograd 연산자를
정의할 수 있습니다. 그 후, 인스턴스(instance)를 생성하고 함수처럼 호출하여
입력 데이터를 포함하는 Variable을 전달하는 식으로 새로운 autograd 연산자를 쉽게
사용할 수 있습니다.

이 예제에서는 ReLU 비선형성(nonlinearity)을 수행하기 위한 사용자 정의 autograd
함수를 정의하고, 2-계층 신경망에 이를 적용해보도록 하겠습니다:

.. includenodoc:: /beginner/examples_autograd/two_layer_net_custom_function.py

TensorFlow: 정적 그래프(Static Graph)
-------------------------------------

PyTorch autograd는 Tensorflow와 많이 닮아보입니다: 두 프레임워크 모두 연산 그래프를
정의하며, 자동 미분을 사용하여 변화도를 계산합니다. 두 프레임워크의 가장 큰 차이점은
Tensorflow의 연산 그래프는 **정적** 이며, PyTorch는 **동적** 연산 그래프를 사용한다는
것입니다.

Tensorflow에서는 연산 그래프를 한 번 정의한 후 동일한 그래프를 계속해서 실행하며
가능한 다른 입력 데이터를 전달합니다. PyTorch에서는 각각의 순전파 단계에서 새로운
연산 그래프를 정의합니다.

정적 그래프는 먼저(Up-front) 그래프를 최적화할 수 있기 때문에 좋습니다; 예를 들어
프레임워크가 효율을 위해 일부 그래프 연산을 합치거나, 여러 GPU나 시스템(machine)에
그래프를 배포하는 전략을 제시할 수 있습니다. 만약 동일한 그래프를 계속 재사용하면,
같은 그래프가 반복되면서 비싼(Costly) 최적화 비용을 잠재적으로 상환할 수 있습니다.

정적 그래프와 동적 그래프는 제어 흐름(Control flow) 측면에서도 다릅니다. 어떤
모델에서 각 데이터 지점(Point)마다 다른 연산 연산을 수행하고 싶을 수 있습니다;
예를 들어 순환 신경망에서 각각의 데이터 지점마다 서로 다른 횟수만큼 펼칠(Unroll)
수 있습니다; 이러한 펼침은 반복문(Loop)으로 구현할 수 있습니다. 정적 그래프에서
반복문은 그래프의 일부가 돼야 합니다; 이러한 이유에서 Tensorflow는 그래프 내에
반복문을 포함하기 위해 ``tf.scan`` 과 같은 연산자를 제공합니다. 동적 그래프에서는
이러한 상황이 더 단순(Simple)해집니다: 각 예제에 대한 그래프를 즉석(on-the-fly)에서
작성하기 때문에, 일반적인 명령형(Imperative) 제어 흐름을 사용하여 각각의 입력에 따라
다른 계산을 수행할 수 있습니다.

위의 PyTorch autograd 예제와는 대조적으로, TensorFlow를 사용하여 간단한 2-계층
신경망을 구성하겠습니다:

.. includenodoc:: /beginner/examples_autograd/tf_two_layer_net.py

`nn` 모듈
===========

PyTorch: nn
-----------

연산 그래프와 autograd는 복잡한 연산자를 정의하고 도함수(derivative)를 자동으로
계산하는데 매우 강력한 패러다임(paradigm)입니다; 하지만 규모가 큰 신경망에서는
autograd 그 자체만으로는 너무 낮은 수준(low-level)일 수 있습니다.

신경망을 구성할 때 종종 연산을 여러 **계층** 으로 배열하는 것으로 생각하게 되는데,
이 중 일부는 학습 도중 최적화가 될 **학습 가능한 매개변수** 를 갖고 있습니다.

Tensorflow에서 `Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
나 `TFLearn <http://tflearn.org/>`__ 같은 패키지는 원초적(raw)인 연산 그래프보다
더 높은 수준의 추상화(higher-level abstraction)를 제공하여 신경망을 구축하는데
있어 유용합니다.

PyTorch에서는 ``nn`` 패키지가 동일한 목적으로 제공됩니다. ``nn`` 패키지는
대략 신경망 계층들과 동일한 **모듈** 의 집합을 정의합니다.
모듈은 입력 Variable을 받고 출력 Variable을 계산하는 한편, 학습 가능한
매개변수를 포함하는 Variable과 같은 내부 상태(internal state)를 갖습니다.
또한, ``nn`` 패키지는 신경망을 학습시킬 때 주로 사용하는 유용한 손실 함수들도
정의합니다.

이번 예제에서는 ``nn`` 패키지를 이용하여 2-계층 신경망을 구성해보겠습니다:

.. includenodoc:: /beginner/examples_nn/two_layer_net_nn.py

PyTorch: optim
--------------

Up to this point we have updated the weights of our models by manually
mutating the ``.data`` member for Variables holding learnable
parameters. This is not a huge burden for simple optimization algorithms
like stochastic gradient descent, but in practice we often train neural
networks using more sophisticated optimizers like AdaGrad, RMSProp,
Adam, etc.

The ``optim`` package in PyTorch abstracts the idea of an optimization
algorithm and provides implementations of commonly used optimization
algorithms.

In this example we will use the ``nn`` package to define our model as
before, but we will optimize the model using the Adam algorithm provided
by the ``optim`` package:

.. includenodoc:: /beginner/examples_nn/two_layer_net_optim.py

PyTorch: Custom nn Modules
--------------------------

Sometimes you will want to specify models that are more complex than a
sequence of existing Modules; for these cases you can define your own
Modules by subclassing ``nn.Module`` and defining a ``forward`` which
receives input Variables and produces output Variables using other
modules or other autograd operations on Variables.

In this example we implement our two-layer network as a custom Module
subclass:

.. includenodoc:: /beginner/examples_nn/two_layer_net_module.py

PyTorch: Control Flow + Weight Sharing
--------------------------------------

As an example of dynamic graphs and weight sharing, we implement a very
strange model: a fully-connected ReLU network that on each forward pass
chooses a random number between 1 and 4 and uses that many hidden
layers, reusing the same weights multiple times to compute the innermost
hidden layers.

For this model we can use normal Python flow control to implement the loop,
and we can implement weight sharing among the innermost layers by simply
reusing the same Module multiple times when defining the forward pass.

We can easily implement this model as a Module subclass:

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

Examples
========

You can browse the above examples here.

Tensors
-------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_tensor/two_layer_net_numpy
   /beginner/examples_tensor/two_layer_net_tensor

.. galleryitem:: /beginner/examples_tensor/two_layer_net_numpy.py

.. galleryitem:: /beginner/examples_tensor/two_layer_net_tensor.py

.. raw:: html

    <div style='clear:both'></div>

Autograd
--------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_autograd/two_layer_net_autograd
   /beginner/examples_autograd/two_layer_net_custom_function
   /beginner/examples_autograd/tf_two_layer_net


.. galleryitem:: /beginner/examples_autograd/two_layer_net_autograd.py

.. galleryitem:: /beginner/examples_autograd/two_layer_net_custom_function.py

.. galleryitem:: /beginner/examples_autograd/tf_two_layer_net.py

.. raw:: html

    <div style='clear:both'></div>

`nn` module
-----------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_nn/two_layer_net_nn
   /beginner/examples_nn/two_layer_net_optim
   /beginner/examples_nn/two_layer_net_module
   /beginner/examples_nn/dynamic_net


.. galleryitem:: /beginner/examples_nn/two_layer_net_nn.py

.. galleryitem:: /beginner/examples_nn/two_layer_net_optim.py

.. galleryitem:: /beginner/examples_nn/two_layer_net_module.py

.. galleryitem:: /beginner/examples_nn/dynamic_net.py

.. raw:: html

    <div style='clear:both'></div>
