C 언어로 PyTorch 확장 기능(custom extension) 만들기
===================================================
**Author**: `Soumith Chintala <http://soumith.ch>`_
  **번역**: `박정환 <http://github.com/9bow>`_


1단계. C 코드 준비하기
---------------------------

먼저, C로 함수를 작성합니다.

다음은 입력들을 모두 더하는 모듈의 순전파 및 역전파 함수의 예제 모듈 구현입니다.

``.c`` 파일에서 ``#include <TH/TH.h>`` 지시자(directive)로 TH를 불러온 뒤,
``#include <THC/THC.h>`` 로 THC도 불러옵니다.

ffi util이 빌드 시에 컴파일러가 이것들을 찾을 수 있도록 할 것입니다.

.. code:: C

    /* src/my_lib.c */
    #include <TH/TH.h>

    int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
    THFloatTensor *output)
    {
        if (!THFloatTensor_isSameSizeAs(input1, input2))
            return 0;
        THFloatTensor_resizeAs(output, input1);
        THFloatTensor_cadd(output, input1, 1.0, input2);
        return 1;
    }

    int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
    {
        THFloatTensor_resizeAs(grad_input, grad_output);
        THFloatTensor_fill(grad_input, 1);
        return 1;
    }


코드를 작성할 때는, Python에서 호출할 모든 함수들을 하나의 헤더 파일로 만들어두는 것 말고는
별도로 유의해야 할 사항(constraint)은 없습니다.

이 헤더 파일은 ffi util이 래퍼(wrapper)를 생성할 때 사용합니다.

.. code:: C

    /* src/my_lib.h */
    int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output);
    int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

이제 다음과 같은 짧은 파일을 이용하여 빌드해보겠습니다:

.. code:: python

    # build.py
    from torch.utils.ffi import create_extension
    ffi = create_extension(
    name='_ext.my_lib',
    headers='src/my_lib.h',
    sources=['src/my_lib.c'],
    with_cuda=False
    )
    ffi.build()

2단계. Python에서 불러오기
--------------------------

위 코드를 실행하면, PyTorch가 ``_ext`` 디렉토리 밑에 ``my_lib`` 디렉토리를
생성할 것입니다.

Package name can have an arbitrary number of packages preceding the
final module name (including none). 빌드가 완료되면 일반적인 Python 파일처럼
불러와서 사용할 수 있습니다.

.. code:: python

    # functions/add.py
    import torch
    from torch.autograd import Function
    from _ext import my_lib


    class MyAddFunction(Function):
        def forward(self, input1, input2):
            output = torch.FloatTensor()
            my_lib.my_lib_add_forward(input1, input2, output)
            return output

        def backward(self, grad_output):
            grad_input = torch.FloatTensor()
            my_lib.my_lib_add_backward(grad_output, grad_input)
            return grad_input

.. code:: python

    # modules/add.py
    from torch.nn import Module
    from functions.add import MyAddFunction

    class MyAddModule(Module):
        def forward(self, input1, input2):
            return MyAddFunction()(input1, input2)


.. code:: python

    # main.py
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from modules.add import MyAddModule

    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__()
            self.add = MyAddModule()

        def forward(self, input1, input2):
            return self.add(input1, input2)

    model = MyNetwork()
    input1, input2 = Variable(torch.randn(5, 5)), Variable(torch.randn(5, 5))
    print(model(input1, input2))
    print(input1 + input2)


