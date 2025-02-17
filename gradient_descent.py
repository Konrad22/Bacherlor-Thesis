import torch
from torch import Tensor
from torch import optim
from composition import einsum_compose
import matplotlib.pyplot as plt
from synthetic import Pattern, gen_synthetic
import tensorly as tl
from time import time
from loss_function import loss_function

#TUCKER DECOMPOSITION: komprimierter core hat kleiner oder gleich große dim wie der correcte Tensor, beachten!

def test_tensor_pattern(number_per_pattern_per_dim, min_number_dim, limit_number_dim, min_size_dim) -> list:
    tensor_list = []
    pattern = [Pattern.Ball, Pattern.Checkers, Pattern.Cuboid, Pattern.Diamond, Pattern.Swiss]
    for pattern in pattern:
        for i in range(limit_number_dim - min_number_dim):
            dim = min_number_dim + i            
            for j in range(number_per_pattern_per_dim):
                size_dim = min_size_dim + j
                dim_list = [size_dim] * dim
                
                tensor_list.append(gen_synthetic(pattern, dim_list))
    return tensor_list

###def compare_einsum_tucker(tensor_core: Tensor, projection_factors: list) -> list:
    time_list = [0, 0]
    tensors = [0, 0]

    start_compose = time()
    compose_tensor = einsum_compose(tensor_core, projection_factors)
    end_compose = time()
    
    start_tucker = time()
    tucker = tl.tucker_to_tensor((tensor_core, projection_factors))
    end_tucker = time()

    time_list[0] = end_compose - start_compose
    time_list[1] = end_tucker - start_tucker
    tensors[0] = compose_tensor
    tensors[1] = tucker
    return [time, tensors]

###def compare_gradient_descent_ALS(correct_tensor: Tensor, core_shape: list, learnrate, momentum, min_loss_change=1e-2) -> list:
    time_list = [0, 0]
    iterations = [0, 0]
    loss = [0, 0]
    start_descent = time()
    gradient = gradient_descent_cut_off_no_fixed_iteration(correct_tensor, core_shape, learnrate, momentum, min_loss_change)
    end_descent = time()
    
    start_als = time()
    #als doing things with the provided input
    end_als = time()

    time_list[0] = end_descent - start_descent
    time_list[1] = end_descent - start_descent
    loss = gradient[0]
    #loss[1] =
    iterations[0] = gradient[1]
    #iterations[1] =
    return [time_list, loss, iterations]


def gradient_descent_cut_off_no_fixed_iteration(correct_tensor: Tensor, core_shape: list, learnrate, momentum, min_loss_change=1e-2) -> list:
    torch.manual_seed(0) #works #only for testing purposes 
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
        #print(loss_new, 3)
        #print(loss_old, 4)
        #print(loss_change, 5)
        #print(min_loss_change, 7)
        if loss_change <= min_loss_change:
            result = [loss_new, i, core, projections, composed_tensor]
        loss_new.backward()
        loss_history.append(loss_new.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    print(result[0], result[1])
    return result

#up to here

def gradient_descent_vary_learnrate_and_momentum_and_iterations_core_size_input(correct_tensor: Tensor, core_shape: list, learnrate, momentum, number_iterations) -> list:
    torch.manual_seed(0) #works #only for testing purposes 
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
    #core_shape = []
    #for i in range(correct_tensor.dim()):
    #    core_shape.append((torch.randint(max(1, correct_tensor.size(dim=i)-4), correct_tensor.size(dim=i), (1, ))).item())
    core = torch.randn(core_shape).requires_grad_()
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, i, core, projections, einsum_compose(core, projections)] #CHANGED
    optimizer = optim.SGD((core, *projections), lr = learnrate, momentum = momentum)
    #number_iterations = 200
    for i in range(number_iterations):       
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)        
        loss = loss_function(correct_tensor, composed_tensor) 
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, i, core, projections, composed_tensor]
            if loss < 1:
                print(i,loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    #plt.show()                                      #works
    print(best_result[0], best_result[1])
    return best_result

def gradient_descent_vary_learnrate_and_momentum_and_iterations_core_smaller_all_zeros(correct_tensor: Tensor, learnrate, momentum, number_iterations) -> list:
    torch.manual_seed(0) #works #only for testing purposes 
    projections = []
    loss_history = []
    core_dim = []
    for i in range(correct_tensor.dim()):
        core_dim.append((torch.randint(max(1, correct_tensor.size(dim=i)-4), correct_tensor.size(dim=i), (1, ))).item())
    core = torch.zeros(core_dim).requires_grad_()
    for i in range(core.dim()):
        factor = torch.zeros(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, i, core, projections, einsum_compose(core, projections)] #CHANGED
    optimizer = optim.SGD((core, *projections), lr = learnrate, momentum = momentum)
    #number_iterations = 200
    for i in range(number_iterations):       
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)        
        loss = loss_function(correct_tensor, composed_tensor) 
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections, composed_tensor]
            if loss < 1:
                print(i,loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    print(best_result[0], best_result[1])
    return best_result

def gradient_descent_vary_learnrate_and_momentum_and_iterations_core_smaller(correct_tensor: Tensor, learnrate, momentum, number_iterations) -> list:
    torch.manual_seed(0) #works #only for testing purposes 
    projections = []
    loss_history = []
    core_dim = []
    for i in range(correct_tensor.dim()):
        core_dim.append((torch.randint(max(1, correct_tensor.size(dim=i)-4), correct_tensor.size(dim=i), (1, ))).item())
    core = torch.randn(core_dim).requires_grad_()
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, i, core, projections, einsum_compose(core, projections)] #CHANGED
    optimizer = optim.SGD((core, *projections), lr = learnrate, momentum = momentum)
    #number_iterations = 200
    for i in range(number_iterations):       
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)        
        loss = loss_function(correct_tensor, composed_tensor) 
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, i, core, projections, composed_tensor]
            if loss < 1:
                print(i,loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    #plt.show()                                      #works
    print(best_result[0], best_result[1])
    return best_result

def gradient_descent_vary_learnrate_and_momentum_and_iterations_core_size_correct(correct_tensor: Tensor, learnrate, momentum, number_iterations) -> list:
    torch.manual_seed(0) #works #only for testing purposes 
    projections = []
    loss_history = []
    core = torch.randn(correct_tensor.shape).requires_grad_()
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, i, core, projections, einsum_compose(core, projections)] #CHANGED
    optimizer = optim.SGD((core, *projections), lr = learnrate, momentum = momentum)
    #number_iterations = 200
    for i in range(number_iterations):       
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)        
        loss = loss_function(correct_tensor, composed_tensor) 
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, i, core, projections, composed_tensor]
            if loss < 1:
                print(i,loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    #plt.show()                                      #works
    print(best_result[0], best_result[1])
    return best_result

def gradient_descent_vary_learnrate_and_momentum_and_iterations_small_core(correct_tensor: Tensor, learnrate, momentum, number_iterations) -> list:
    torch.manual_seed(0) #works #only for testing purposes 
    projections = []
    loss_history = []
    core = torch.randn(correct_tensor.shape).requires_grad_()
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, core, projections, einsum_compose(core, projections)] #CHANGED
    optimizer = optim.SGD((core, *projections), lr = learnrate, momentum = momentum)
    #number_iterations = 200
    for i in range(number_iterations):       
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)        
        loss = loss_function(correct_tensor, composed_tensor) 
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections, composed_tensor]
            if loss < 1:
                print(i,loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    print(best_result[0])
    return best_result

def gradient_descent_core_equal_correct_size(correct_tensor: Tensor) -> list:
    torch.manual_seed(0) #works #only for testing purposes 
    projections = []
    loss_history = []
    core = torch.randn(correct_tensor.shape).requires_grad_()
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, core, projections, einsum_compose(core, projections)] #CHANGED
    optimizer = optim.SGD((core, *projections), lr = 1e-4, momentum = 0.8)
    number_iterations = 200
    for i in range(number_iterations):       
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)        
        loss = loss_function(correct_tensor, composed_tensor) 
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections, composed_tensor]
            if loss < 1:
                print(i,loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    print(best_result[0])
    return best_result

def gradient_descent_core_random_size(type: str, correct_tensor: Tensor) -> list:
    #dim = len(correct_tensor.shape)
    #core_tensor = torch.randn(correct_tensor.shape)
    #list with randn tensors for the initialzing of projection tensors
    #optimizer = type
    torch.manual_seed(0) #works #only for testing purposes
    #number_dim = torch.randint(4, 10, (1, ))
    #correct_tensor_dim = torch.randint(2, 10, (number_dim.item(), ))
    #correct_tensor = torch.randn(correct_tensor_dim.tolist())
    #print(correct_tensor.size(), 'correct')
    #min_loss = 2000
    projections = []
    #best_result =[min_loss, 0, 0]
    #print(correct_tensor)
    loss_history = []
    core_dim = []
    for i in range(correct_tensor.dim()):
        core_dim.append((torch.randint(1, correct_tensor.size(dim=i), (1, ))).item())
    core = torch.randn(core_dim).requires_grad_()
    #print(core.size(), 'core')
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
        #print(factor.size())
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, core, projections, einsum_compose(core, projections)] #CHANGED
    #print(core, projections)
    match type:
        case 'SDG':
            optimizer = optim.SGD((core, *projections), lr = 1e-4, momentum = 0.8)
        
        case 'ASDG':
            optimizer = optim.ASGD((core, *projections), lr = 1e-3)

        case _:
            print('Not a valid optimizer.')
            pass

    #optimizer = optim.SGD((core, *projections), lr = 1e-3, momentum = 0.9)
    number_iterations = 200
    for i in range(number_iterations):
        #print(i)
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        #print(composed_tensor.shape)
        #print(correct_tensor.shape)
        loss = loss_function(correct_tensor, composed_tensor)
        #print(composed_tensor.size())
        #print(i,loss)
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections, composed_tensor]
            if loss < 1:
                print(i,loss)
        #if i>999/1000*number_iterations:
        #    print(loss)
        #print(loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    #print(loss_history[7])
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    #print(1)
    #print(compose(best_result[1], best_result[2]) == best_result[3])
    #print(loss_function(correct_tensor, compose(best_result[1], best_result[2]))) why doesnt this work? ask maurice
    #print(2)
    print(best_result[0])
    #print(correct_tensor, composed_tensor)
    return best_result
    

#def averaged_stoch_gradient_descent(correct_tensor: Tensor) -> list:
#    gradient_descent('ASGD', correct_tensor)
#    return

#def averaged_stoch_gradient_descent(correct_tensor: Tensor) -> list:
 #   gradient_descent('SGD', correct_tensor)
 #   return

def gradient_descent_randomize_correct_tensor_shape_dim_random() -> list:
    torch.manual_seed(0) #works #only for testing purposes
    number_dim = torch.randint(4, 10, (1, ))
    correct_tensor_dim = torch.randint(2, 10, (number_dim.item(), ))
    correct_tensor = torch.randn(correct_tensor_dim.tolist())
    #print(correct_tensor.size(), 'correct')
    #min_loss = loss_function(correct_tensor, compose(core, projections)) #CHANGED
    projections = []
    #best_result =[min_loss, 0, 0] #CHANGED
    #print(correct_tensor)
    loss_history = []
    core_dim = torch.randint(2, 10, (correct_tensor.dim(), ))
    core = torch.randn(core_dim.tolist()).requires_grad_()
    #print(core.size(), 'core')
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
        #print(factor.size())
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, core, projections, einsum_compose(core, projections)] #CHANGED
    #print(core, projections)
    optimizer = optim.SGD((core, *projections), lr = 1e-3, momentum = 0.9)
    number_iterations = 10
    for i in range(number_iterations):
        #print(i)
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        #print(composed_tensor.shape)
        #print(correct_tensor.shape)
        loss = loss_function(correct_tensor, composed_tensor)
        #print(composed_tensor.size())
        #print(i,loss)
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections, composed_tensor] #how do I protect this core, projections being updated by the optimzier step?
            #print(i, best_result[0], loss_function(correct_tensor, composed_tensor), loss_function(correct_tensor, compose(best_result[1], best_result[2])))
            if loss < 1:
                print(i,loss)
        #if i>999/1000*number_iterations:
        #    print(loss)
        #print(loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    #print(1)
    #print(compose(best_result[1], best_result[2]) == best_result[3])
    #print(3)
    #print(loss_function(correct_tensor, compose(best_result[1], best_result[2]))) #why doesnt this work? ask maurice
    #print(2)
    #print(best_result[0])
    #print(correct_tensor, composed_tensor)
    return best_result

def gradient_descent_randomize_correct_tensor_shape_dim_three() -> list:
    torch.manual_seed(0) #works #only for testing purposes
    correct_tensor_dim = torch.randint(0, 10, (3, ))
    correct_tensor = torch.randn(correct_tensor_dim.tolist())
    print(correct_tensor.size())
    #min_loss = 20 #CHANGED
    projections = []
    #best_result =[min_loss, 0, 0] #CHANGED
    #print(correct_tensor)
    loss_history = []
    core_dim = torch.randint(0, 10, (correct_tensor.dim(), ))
    core = torch.randn(core_dim.tolist()).requires_grad_()
    print(core.size())
    null_list = []
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
        null_list.append(torch.zeros(correct_tensor.size(dim=i), core.size(dim=i)))
        print(factor.size())
    min_loss = loss_function(correct_tensor, einsum_compose(core, projections)) #CHANGED
    best_result =[min_loss, core, projections, einsum_compose(core, projections)] #CHANGED
    print(min_loss)
    #print(core, projections)
    optimizer = optim.SGD((core, *projections), lr = 1e-3, momentum = 0.9)
    #print(correct_tensor)
    null_list = []
    number_iterations = 22
    for i in range(number_iterations):
        #print(i)
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        #print(composed_tensor.shape)
        #print(correct_tensor.shape)
        loss = loss_function(correct_tensor, composed_tensor)
        if loss < min_loss:
            min_loss = loss
            best_core = 0 + core
            best_projections = null_list + projections
            best_result = [min_loss, best_core, best_projections, composed_tensor] #why does that not work?
            #print(i, best_result[0], loss_function(correct_tensor, composed_tensor), loss_function(correct_tensor, compose(best_result[1], best_result[2])))
            #print(core == best_result[1], projections == best_result[2])
            #print(core, projections)
            if loss < 1:
                print(i,loss)
        #if i>999/1000*number_iterations:
        #    print(loss)
        #print(loss)
        loss.backward()
        #print(10, loss_function(correct_tensor, compose(core, projections)))
        loss_history.append(loss.item()) #works
        #print(11, loss_function(correct_tensor, compose(core, projections)))
        optimizer.step()
        #print(core == best_result[1], projections == best_result[2])
        #print(12, loss_function(correct_tensor, compose(core, projections)))

    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    print(1)
    #print(compose(best_result[1], best_result[2]) == best_result[3])
    #print(correct_tensor)
    print(loss_function(correct_tensor, einsum_compose(best_result[1], best_result[2]))) #why doesnt this work? ask maurice
    print(2)
    print(best_result[0])
    print(3)
    #print(loss_history)
    #print(core, projections)
    #print(correct_tensor, composed_tensor)
    return

def gradient_descent_projection_size_based_on_core_and_correct_and_core_shape_is_random() -> list:
    torch.manual_seed(0) #works #only for testing purposes
    correct_tensor = torch.randn(3,4,5)
    min_loss = 20
    projections = []
    best_result =[min_loss, 0, 0]
    #print(correct_tensor)
    loss_history = []
    core_dim = torch.randint(0, 10, (correct_tensor.dim(), ))
    core = torch.randn(core_dim.tolist()).requires_grad_()
    print(core.size())
    for i in range(core.dim()):
        factor = torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_()
        projections.append(factor)
        print(factor.size())
    #print(core, projections)
    optimizer = optim.SGD((core, *projections), lr = 1e-3, momentum = 0.9)
    number_iterations = 10000
    for i in range(number_iterations):
        #print(i)
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        #print(composed_tensor.shape)
        #print(correct_tensor.shape)
        loss = loss_function(correct_tensor, composed_tensor)
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections, composed_tensor]
            if loss < 6:
                print(i,loss)
        #if i>999/1000*number_iterations:
        #    print(loss)
        #print(loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    #print(1)
    #print(compose(best_result[1], best_result[2]) == best_result[3])
    #print(loss_function(correct_tensor, compose(best_result[1], best_result[2]))) why doesnt this work? ask maurice
    #print(2)
    #print(best_result[0])
    #print(correct_tensor, composed_tensor)
    return

def gradient_descent_projection_size_based_on_core_and_core_shape_is_correct_tensor() -> list:
    torch.manual_seed(0) #works #only for testing purposes
    correct_tensor = torch.randn(3,4,5)
    min_loss = 20
    projections = []
    best_result =[min_loss, 0, 0]
    #print(correct_tensor)
    loss_history = []
    core = torch.randn(correct_tensor.shape).requires_grad_()
    for i in range(core.dim()):
        projections.append(torch.randn(correct_tensor.size(dim=i), core.size(dim=i)).requires_grad_())
    print(core, projections)
    optimizer = optim.SGD((core, *projections), lr = 1e-3, momentum = 0.9)
    number_iterations = 10000
    for i in range(number_iterations):
        #print(i)
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        #print(composed_tensor.shape)
        #print(correct_tensor.shape)
        loss = loss_function(correct_tensor, composed_tensor)
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections, composed_tensor]
            if loss < 0.03:
                print(i,loss)
        #if i>999/1000*number_iterations:
        #    print(loss)
        #print(loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    #print(1)
    #print(compose(best_result[1], best_result[2]) == best_result[3])
    #print(loss_function(correct_tensor, compose(best_result[1], best_result[2]))) why doesnt this work? ask maurice
    #print(2)
    #print(best_result[0])
    #print(correct_tensor, composed_tensor)
    return

def gradient_descent_test_fixed_size() -> list:
    torch.manual_seed(0) #works #only for testing purposes
    correct_tensor = torch.randn(3,4,5)
    min_loss = 20
    best_result =[min_loss, 0, 0]
    #print(correct_tensor)
    loss_history = []
    core = torch.randn(correct_tensor.shape).requires_grad_()
    projections = [torch.randn(3,3).requires_grad_(), torch.randn(4,4).requires_grad_(), torch.randn(5,5).requires_grad_()]
    #print(core, projections)
    optimizer = optim.SGD((core, *projections), lr = 1e-3, momentum = 0.9)
    number_iterations = 10000
    for i in range(number_iterations):
        #print(i)
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        #print(composed_tensor.shape)
        #print(correct_tensor.shape)
        loss = loss_function(correct_tensor, composed_tensor)
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections]
            if loss < 0.05:
                print(i,loss)
        if i>999/1000*number_iterations:
            print(loss)
        #print(loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    #print(correct_tensor, composed_tensor)
    return

def gradient_descent_test_fixed_core() -> list:
    torch.manual_seed(0) #works #only for testing purposes
    correct_tensor = torch.randn(3,4,5).fill_(3)
    min_loss = 20
    best_result =[min_loss, 0, 0]
    #print(correct_tensor)
    loss_history = []
    core = torch.randn(correct_tensor.shape).requires_grad_()
    projections = [torch.randn(3,3).requires_grad_(), torch.randn(4,4).requires_grad_(), torch.randn(5,5).requires_grad_()]
    #print(core, projections)
    optimizer = optim.SGD((core, *projections), lr = 1e-3, momentum = 0.9)
    number_iterations = 10000
    for i in range(number_iterations):
        #print(i)
        optimizer.zero_grad()
        composed_tensor = einsum_compose(core, projections)
        #print(composed_tensor.shape)
        #print(correct_tensor.shape)
        loss = loss_function(correct_tensor, composed_tensor)
        if loss < min_loss:
            min_loss = loss
            best_result = [min_loss, core, projections]
            if loss < 1:
                print(i,loss)
        if i>999/1000*number_iterations:
            print(loss)
        #print(loss)
        loss.backward()
        loss_history.append(loss.item()) #works
        optimizer.step()
    plt.plot(range(len(loss_history)), loss_history) #works
    plt.show()                                      #works
    #print(correct_tensor, composed_tensor)
    return

def main():
    #gradient_descent_test_fixed_everything()
    #gradient_descent_test_fixed_size()
    #gradient_descent_projection_size_based_on_core_and_core_shape_is_correct_tensor()
    #gradient_descent_projection_size_based_on_core_and_correct_and_core_shape_is_random()
    #gradient_descent_randomize_correct_tensor_shape_dim_three()
    #gradient_descent_randomize_correct_tensor_shape_dim_random()
    torch.manual_seed(0)
    number_dim = torch.randint(4, 8, (1, ))
    correct_tensor_dim = torch.randint(4, 7, (number_dim.item(), ))
    #print(number_dim)
    #print(correct_tensor_dim)
    #correct_tensor = torch.randn(correct_tensor_dim.tolist())
    #correct_tensor = torch.randint(2, 5, (3, 4, 4, 5))
    #pattern_tensor = test_tensor_pattern(5, 3, 7, 6)
    correct_tensor = [torch.full((8, 8, 8, 8), 5), torch.full((4,5,6,7), 5), torch.full((5,4,5,4,5,4,5),2), gen_synthetic(Pattern.Ball, (4,4,4,4)), gen_synthetic(Pattern.Checkers, (4,4,4,4)), 
    gen_synthetic(Pattern.Cuboid, (4,4,4,4)), gen_synthetic(Pattern.Diamond, (4,4,4,4)), gen_synthetic(Pattern.Swiss, (4,4,4,4))]
    correct_tensor = torch.full((8, 8, 8, 8), 5)
    #correct_tensor2 = gen_synthetic(Pattern.Diamond, (8,8,8,8))
    correct_tensor2 = torch.randint(2, 5, (6, 9, 4, 7))
    #print(correct_tensor[3].shape)
    #gradient_descent_core_random_size('SDG', correct_tensor)
    #gradient_descent_core_equal_correct_size(correct_tensor)
    #gradient_descent_core_random_size('ASDG', correct_tensor)
    #gradient_descent_vary_learnrate_and_momentum_and_iterations_core_size_correct(correct_tensor, 1e-3, 0.8, 400)
    #gradient_descent_vary_learnrate_and_momentum_and_iterations_core_smaller(correct_tensor, 1e-3, 0.8, 400)
    #gradient_descent_vary_learnrate_and_momentum_and_iterations_core_smaller_all_zeros(correct_tensor, 1e-3, 0.9, 300)

    


    core_shape = []
    for i in range(correct_tensor.dim()):
        core_shape.append((torch.randint(max(1, correct_tensor.size(dim=i)-4), correct_tensor.size(dim=i), (1, ))).item())
        #core_shape.append(1)
    print(core_shape)

    core_shape2 = []
    for i in range(correct_tensor2.dim()):
        core_shape2.append((torch.randint(max(1, correct_tensor2.size(dim=i)-4), correct_tensor2.size(dim=i), (1, ))).item())

    #gradient_descent_cut_off_no_fixed_iteration(correct_tensor, core_shape, 1e-3, 0.8)

#def other():
    lr_range = 3
    momentum_range = 20
    iterations_range = 1
    lr = [1e-2/(10**x) for x in range(lr_range)]
    momentum = [y/20 for y in range(momentum_range)]
    number_iterations = [2000 + z*200 for z in range(iterations_range)]
    table_texts = []

    #for tensor in correct_tensor:
    for z in range(iterations_range):
        cell_text = []
        for x in range(lr_range):
            row = []
            for y in range(momentum_range):                  
                row.append('%1.3f' % gradient_descent_vary_learnrate_and_momentum_and_iterations_core_size_input(correct_tensor, core_shape, lr[x], momentum[y], number_iterations[z])[0])
            cell_text.append(row)
        table_texts.append(cell_text)
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table = plt.table(cellText=table_texts[z], rowLabels=lr, colLabels=momentum, loc='center')
        plt.title('%d' % number_iterations[z])
        fig.tight_layout()      
        plt.savefig(f"plot_{z}_momentum_range_{momentum_range}_half.png", dpi=300)

    table_texts2 = []
    for z in range(iterations_range):
        cell_text2 = []
        for x in range(lr_range):
            row = []
            for y in range(momentum_range):                  
                row.append('%1.3f' % gradient_descent_vary_learnrate_and_momentum_and_iterations_core_size_input(correct_tensor2, core_shape2, lr[x], momentum[y], number_iterations[z])[0])
            cell_text2.append(row)
        table_texts2.append(cell_text2)
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table = plt.table(cellText=table_texts2[z], rowLabels=lr, colLabels=momentum, loc='center')
        plt.title('%d' % number_iterations[z])
        fig.tight_layout()      
        plt.savefig(f"plot_{z}_momentum_range_{momentum_range}_half_2.png", dpi=300)

    # Add a table
    # for i in range(iterations_range):
    #     fig, ax = plt.subplots()
    #     fig.patch.set_visible(False)
    #     ax.axis('off')
    #     ax.axis('tight')
    #     ax.table = plt.table(cellText=table_texts[i], rowLabels=rows, colLabels=momentum, loc='center')
    #     a = (i+1)*100
    #     plt.title('%d' % title[i])
    #     fig.tight_layout()      
    #     plt.show()

    return
#main()

###def test():
    core_tensor = torch.randint(2, 5, (6, 9, 4, 7))
    projection_factors = []
    for i in range(core_tensor.dim()):
        factor = torch.randn(core_tensor.size(dim=i), core_tensor.size(dim=i)+2).requires_grad_()
        projection_factors.append(factor)
    print(compare_einsum_tucker(core_tensor, projection_factors))
    return
#test()