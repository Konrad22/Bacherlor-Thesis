import torch
from torch import Tensor
from torch import optim
from composition import einsum_compose
import matplotlib.pyplot as plt
from loss_function import loss_function

def gradient_descent(correct_tensor: Tensor, core_shape: list, learnrate, momentum, min_loss_change=1e-2) -> list:
    projections = []
    loss_history = []
    wrong = False
   
    if correct_tensor.dim() != len(core_shape):
        wrong = True
    for i in range(correct_tensor.dim()):
        if correct_tensor.size(dim=i) <= core_shape[i]:
            wrong = True
            break
    if wrong == True:
        print(list(correct_tensor.shape))
        print(core_shape)
        print('Wrong core shape.')
        return ['Core shape error']
    core = torch.randn(core_shape).requires_grad_()
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
    loss_new = 0
    loss_old = loss_new + 1 + min_loss_change
    loss_change = abs(loss_old - loss_new)
    i = 0
    result =[loss_new, i, core, projections, einsum_compose(core, projections)]
    optimizer = optim.SGD((core, *projections), lr = learnrate, momentum = momentum)
    while loss_change > min_loss_change:
        i = i + 1       
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        loss_old = loss_new        
        loss_new = loss_function(correct_tensor, composed_tensor)
        loss_change = abs(loss_old - loss_new)
        if loss_change <= min_loss_change:
            result = [loss_new, i, core, projections, composed_tensor]
        loss_new.backward()
        loss_history.append(loss_new.item())
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history)
    plt.show()
    print(result[0], result[1])
    return result