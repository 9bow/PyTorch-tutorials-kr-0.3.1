# -*- coding: utf-8 -*-
"""
멀티-GPU 예제
==================

데이터 병렬 처리(Data Parallelism)는 미니배치를 여러 개의 더 작은 미니배치로 자르고
각각의 작은 미니배치를 병렬적으로 연산하는 것입니다.

데이터 병렬 처리는 ``torch.nn.DataParallel`` 을 사용하여 구현합니다.
``DataParallel`` 로 감쌀(wrap) 수 있는 모듈은 배치 차원(batch dimension)에서
여러 GPU에서 병렬 처리될 것입니다.

DataParallel
-------------
"""
import torch.nn as nn


class DataParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)

        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(20, 20)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

########################################################################
# 코드는 CPU 모드 때와 바뀔 필요가 없습니다.
#
# DataParallel에 대한 문서는 `여기 <http://pytorch.org/docs/nn.html#dataparallel>`_
# 에서 확인하실 수 있습니다.
#
# **DataParallel이 구현된 기본형(Primitive):**
#
#
# 일반적으로, PyTorch의 `nn.parallel` 기본형은 독립적으로 사용할 수 있습니다.
# 간단한 MPI류의 기본형을 구현했습니다:
#
# - 복제(replicate): 여러 기기(Device)에 모듈을 복제합니다.
# - 분산(scatter): 첫번째 차원(First-dimension)에서 입력을 분산합니다.
# - 수집(gather): 첫번째 차원의 입력을 수집하고 연결(Concatenate)합니다.
# - 병렬적용(parallel\_apply): 이미 분산된 입력의 집합을 이미 분산된 모델의
# 집합에 적용합니다.
#
# 더 명확히 알아보기 위해, 이러한 요소를 사용하여 구성한 ``data_parallel``
# 함수가 있습니다.


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)

########################################################################
# 모델의 일부는 CPU, 일부는 GPU에서
# ----------------------------------
#
# 일부는 CPU에서, 일부는 GPU에서 구현한 작은 신경망 예제를 살펴보겠습니다:


class DistributedModel(nn.Module):

    def __init__(self):
        super().__init__(
            embedding=nn.Embedding(1000, 10),
            rnn=nn.Linear(10, 10).cuda(0),
        )

    def forward(self, x):
        # Compute embedding on CPU
        x = self.embedding(x)

        # Transfer to GPU
        x = x.cuda(0)

        # Compute RNN on GPU
        x = self.rnn(x)
        return x

########################################################################
#
# 지금까지 기존 Torch 사용자를 위한 간단한 PyTorch 개요였습니다.
# 배울 것은 아주 많이 있습니다.
#
# ``optim`` 패키지, 데이터 로더 등을 소개하고 있는 더 포괄적인 입문용 튜토리얼을
# 보시기 바랍니다: :doc:`/beginner/deep_learning_60min_blitz`.
#
# 또한, 다음의 내용들도 살펴보세요.
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train an face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _`Deep Learning with PyTorch: a 60-minute blitz`: https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train an face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _다른 예제들 참고하기: https://github.com/pytorch/examples
# .. _더 많은 튜토리얼 보기: https://github.com/pytorch/tutorials
# .. _포럼에서 PyTorch에 대해 얘기하기: https://discuss.pytorch.org/
# .. _Slack에서 다른 사용자와 대화하기: http://pytorch.slack.com/messages/beginner/
