import torch
from torch import Tensor
from composition import einsum_compose
from composition import einsum_n_mode_product
import matplotlib.pyplot as plt
import tensorly as tl
from time import time
from loss_function import loss_function as diff
import numpy as np
from pandas import DataFrame
import pandas as pd

def result_same(tensor_core: Tensor, projection_factors_torch: list, projection_factors_tensorly: list):
    #result_same = (einsum_compose(tensor_core, projection_factors_torch) == tl.tucker_to_tensor((tl.tensor(tensor_core), projection_factors_tensorly)))
    #print(einsum_compose(tensor_core, projection_factors_torch)-tl.tucker_to_tensor((tl.tensor(tensor_core), projection_factors_tensorly)))
    result_same = diff(einsum_compose(tensor_core, projection_factors_torch),tl.tucker_to_tensor((tl.tensor(tensor_core), projection_factors_tensorly)))
    return result_same.item()


def compare_einsum_tucker(tensor_core: Tensor, projection_factors_torch: list, projection_factors_tensorly: list) -> list:
    n_runs = 100
    start_compose = time()
    for _ in range(n_runs):
        #compose_tensor = einsum_compose(tensor_core, projection_factors_torch)
        einsum_compose(tensor_core, projection_factors_torch)
    end_compose = time()
    time_compose = (end_compose - start_compose) / n_runs

    start_tucker = time()
    for _ in range(n_runs):
        #tucker_tensor = tl.tucker_to_tensor((tl.tensor(tensor_core), projection_factors_tensorly))
        tl.tucker_to_tensor((tl.tensor(tensor_core), projection_factors_tensorly))
    end_tucker = time()
    time_tucker = (end_tucker - start_tucker) / n_runs

    if time_tucker > 0:
        return time_compose, time_tucker, time_compose - time_tucker, time_compose/time_tucker
    
    else:
        return time_compose, time_tucker, time_compose - time_tucker, 100

def repeat():
    for i in range(4):
        main(i)
    return

def main(count):
    #mehrere Tensoren vorgeben, als Liste?
    test_tensors = []
    #mehrere Listen von Projektionsfaktoren
    list_projection_factors = []
    #'Matrix' von Tensoren und Faktoren erzeugen
    core_list = []
    #core_3 = [[5,5,5],[4,6,4],[10,10,10],[40, 40, 30],[80,70,50]]
    #core_4 = [[5,5,5,5],[4,6,4,6],[10,10,10,10],[40, 40, 30, 30],[80,70,50,60]]
    #core_5 = [[5,5,5,5,5],[4,6,4,6,4],[10,10,10,10,10],[40, 40, 30, 30, 40],[80,70,50,60,55]]


    primary_core = [[12],[15],[20],[30]]
    #primary_core = [[12],[15]]
    core_3 = [core*3 for core in primary_core]
    core_4 = [core*4 for core in primary_core]
    core_5 = [core*5 for core in primary_core]
    core_6 = [core*6 for core in primary_core]
    #core_10 = [core*10 for core in primary_core]
    #for i in range(len(core_5)):
    #    core_10.append(core_5[i]*2)
        #print(target_5[i]*2)
    core_list = [core_3, core_4, core_5, core_6[:-2]]
    #core_list = [core_3]
    #print(target_10)

    #rename the lists, switch core and target because core < target
    target_list = []
    print('step 1')
    for i in range(len(core_list)):
        target_list.append(core_list[i][1:])
        #print(core_list[i])
        #print(target_list[i][0])
        target_list[i].append((np.array(core_list[i][-1]) * 2).tolist())
        #print(target_list[i])
        #print(core_list[i])
    #print(core_list)
    
    print('step 2')

    #turn core shapes into cores
    for i in range(len(core_list)):
        #print(i)
        for j in range(len(core_list[i])):
            #print(core_list[i][j])
            core_list[i][j] = torch.randn(core_list[i][j])
            

    #core_to_projection = []
    print('step 3, up to here okay')
    #use core instead of target
    '''for h in range(len(core_list)):
        core_to_projection_sorted = []
        for j in range(len(core_list[h])):
            #core_3/4/5/10 == core_list[h]
            projections_torch = []
            projections_tensorly = []
            for i in range(core_list[h][j].dim()):
                #core_3/4/5/10[j].dim() == 3/4/5/10
                #core_3[0] == 12 -> target_3[0] == 15
                #if j < 2:
                #    a = 0
                #else:
                #    a = j - 1
                for k in range(len(target_list[h])):
                    #target_list[h] == target_3/4/5/10
                    #generates projection factors from core to target
                    factor = torch.randn(target_list[h][k].size(dim=i), core_list[h][j].size(dim=i))
                    projections_torch.append(factor)
                    projections_tensorly.append(tl.tensor(factor))
                    #print(factor.shape)
            core_to_projection_sorted = [].append([core_list[h][j], projections_torch, projections_tensorly])
        core_to_projection.append(core_to_projection_sorted)'''

    core_to_projections = []
    for h in range(len(core_list)):
        core_to_projection_h =[]
        for j in range(len(core_list[h])):
            core_to_projection_h_j = [[core_list[h][j]]]
            #print(list(core_list[h][j].shape))
            #first element of the list is the core, use that later
            for k in range(j, len(core_list[h])):          
                projections_torch = []
                projections_tensorly = []
                #projections = [factor1, factor2, ...]
                #print('okay?')
                for i in range(len(core_list[h][j].shape)):
                    #print(len(core_list[h][j].shape))
                    #print(len(target_list[h][k]))
                    #print(target_list[h][k])
                    factor = torch.randn(target_list[h][k][i], core_list[h][j].size(dim=i))
                    projections_torch.append(factor)
                    projections_tensorly.append(tl.tensor(factor))
                core_to_projection_h_j.append([projections_torch, projections_tensorly, target_list[h][k]])
                #core_to_projection_h_j = [[core_list[h][j]], [both_proj], [both_proj],...]
            core_to_projection_h.append(core_to_projection_h_j)
            #core_to_projection_h = [core_to_projection_h_0, core_to_projection_h_1,...]
        core_to_projections.append(core_to_projection_h)
        #core_to_projection = [core_to_projection_0, core_to_projection_1,...]
                

    print('step 4, okay up to here as well')
    #results = []

    '''for core_to_projection in core_to_projections:
        #print(core_projection[0].shape)
        result_size = []
        for i in range(len(core_to_projection)):
            result_size.append(result_same(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]), core_to_projection[0].shape, compare_einsum_tucker(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]))
            #print(result_same(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]))
            #print(core_to_projection[0].shape, compare_einsum_tucker(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]))
            print(result_size)
        results.append(result_size)'''

    core_shapes = []
    target_shapes = []
    error_margins = []
    times_compose = []
    times_tucker = []
    time_diffs = []
    time_factors = []

    '''#core_to_projection_h_j = [[core_list[h][j]], [both_proj], [both_proj],...]
        
        #core_to_projection_h = [core_to_projection_h_0, core_to_projection_h_1,...]
        
        #core_to_projection = [core_to_projection_0, core_to_projection_1,...]'''
    i = 3
    #collecting the information by column instead of rows
    for same_dim in core_to_projections:
        print(f'coresteps = {len(same_dim[0][0][0].shape)}')
        core_shapes_dim = []
        target_shapes_dim = []
        error_margins_dim = []
        times_compose_dim = []
        times_tucker_dim = []
        time_diffs_dim = []
        time_factors_dim = []
        for same_core in same_dim:
            #print(same_core)
            core = same_core[0][0]
            core_shape = list(core.shape)
            print('core', core_shape)
            for factors in same_core[1:]:
                torch_factor = factors[0]
                tensorly_factor = factors[1]
                target_shape = factors[2]
                print('target', target_shape)
                error_margin = result_same(core, torch_factor, tensorly_factor)
                time_compose, time_tucker, time_diff, time_factor = compare_einsum_tucker(core, torch_factor, tensorly_factor)
                core_shapes_dim.append(core_shape)
                target_shapes_dim.append(target_shape)
                error_margins_dim.append(error_margin)
                times_compose_dim.append(time_compose)
                times_tucker_dim.append(time_tucker)
                time_diffs_dim.append(time_diff)
                time_factors_dim.append(time_factor)
        core_shapes.append(core_shapes_dim)
        target_shapes.append(target_shapes_dim)
        error_margins.append(error_margins_dim)
        times_compose.append(times_compose_dim)
        times_tucker.append(times_tucker_dim)
        time_diffs.append(time_diffs_dim)
        time_factors.append(time_factors_dim)
        i = i + 1



        '''#print(core_projection[0].shape)
        result_size = []
        for i in range(len(core_to_projection)):
            result_size.append(result_same(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]), core_to_projection[0].shape, compare_einsum_tucker(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]))
            #print(result_same(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]))
            #print(core_to_projection[0].shape, compare_einsum_tucker(core_to_projection[i][0], core_to_projection[i][1], core_to_projection[i][2]))
            print(result_size)
        results.append(result_size)

        [projections_torch, projections_tensorly, list(core_list[h][j].shape), target_list[h][k]]'''

    print('step 5')

    dims = ['3', '4', '5', '6']

    for i in range(len(core_to_projections)):
        #sheet_number = dims[i]
        df = DataFrame({'Core shape': core_shapes[i], 'Target shape': target_shapes[i], 'Error margins': error_margins[i], 'Time compose': times_compose[i], 
            'Time tucker': times_tucker[i],'Difference time (tucker - compose)': time_diffs[i],'Time ratio (compose/tucker)': time_factors[i]})
        with pd.ExcelWriter('test_results.xlsx', engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=f'dim_{count + 3}_{dims[i]}', index=False)


    '''correct_tensor = torch.randn([30,30,26])
    core = torch.randn([4,5,4])
    core2 = torch.randn([6,7,6])
    core3 = torch.randn([25,25,25])
    #cores = [core, core2, core3]
    cores = [core]
    print(core.shape)
    #projections_torch = []
    #projections_tensorly = []
    core_to_projections = []
    for j in range(len(cores)):
        projections_torch = []
        projections_tensorly = []
        for i in range(cores[j].dim()):
            factor = torch.randn(correct_tensor.size(dim=i), cores[j].size(dim=i))
            projections_torch.append(factor)
            projections_tensorly.append(tl.tensor(factor))
            #print(factor.shape)
        core_to_projections.append([cores[j], projections_torch, projections_tensorly])'''
    #for tensor in test_tensors:
    #    for projection_factors in list_projection_factors:
    #        core_projections = core_projections + [tensor, projection_factors]
    #Paare durchlaufen lassen
    #tl.tucker_to_tensor([tl.tensor(core), projections_tensorly])


    #for core_projection in core_projections:
        #print(core_projection[0].shape)
    #    print(result_same(core_projection[0], core_projection[1], core_projection[2]))
    #    print(core_projection[0].shape, compare_einsum_tucker(core_projection[0], core_projection[1], core_projection[2]))

    #Zeiten sammeln und als Tabelle darstellen? Inspiration vom Paper nehmen
    '''fixed_core = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
    p1 = torch.tensor([[1,4],[2,5],[3,6]])
    p2 = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
    p3 = torch.tensor([[6,5], [4,3], [2,1]])
    print(fixed_core.shape, p1.shape, p2.shape, p3.shape)
    #print(tl.tucker_to_tensor(tl.tensor(fixed_core), tl.tensor(p1)))
    tucker=tl.tenalg.mode_dot(tl.tensor(fixed_core), tl.tensor(p1), 0)
    e = einsum_n_mode_product(fixed_core,p1,0)
    print(tucker == e)
    a = einsum_n_mode_product(fixed_core,p1,0)
    b = einsum_n_mode_product(a, p2, 1)
    c = einsum_n_mode_product(b, p3, 2)
    d = einsum_compose(fixed_core, [p1,p2,p3])
    print(c == d)
    print(result_same(fixed_core, *[[p1,p2,p3],[tl.tensor(p1), tl.tensor(p2), tl.tensor(p3)]]))'''
    return

#main()
repeat()