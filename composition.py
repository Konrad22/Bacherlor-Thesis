import torch
from torch import Tensor
from symbols import SymbolGenerator

def einsum_compose(tensor_core: Tensor, projection_factors: list) -> Tensor:
    if len(tensor_core.shape) != len(projection_factors):
        print('error')
        return
    symbol_generator = SymbolGenerator()
    core_symbols = symbol_generator.generate(len(tensor_core.shape))
    result_symbols = symbol_generator.generate(len(projection_factors))
    zipped_symbols = zip(result_symbols, core_symbols)
    
    projection_symbols = []

    for m, n in zipped_symbols:
        projection_symbols = projection_symbols + [str(m)+str(n)]
    
    #tensor_list = projection_factors.insert(0, tensor_core)
    tensor_composed = torch.einsum(f"{core_symbols}, {', '.join(projection_symbols)} -> {result_symbols}", tensor_core, *projection_factors)
    return tensor_composed

def einsum_n_mode_product(tensor: Tensor, matrix: Tensor, mode: int) -> Tensor:
    if mode >= len(tensor.shape):
        print('wrong mode, %d > %d' % (mode, len(tensor.shape - 1)))
        return
    if tensor.size(dim = mode) != matrix.size(dim=1):
        print('wrong mode, tensor dim %d != %d' % (tensor.size(dim = mode), matrix.size(dim=0)))
    symbol_generator = SymbolGenerator()
    tensor_symbols = symbol_generator.generate(len(tensor.shape))
    matrix_symbols = symbol_generator.generate(1) + tensor_symbols[mode]
    result_symbols = tensor_symbols[:mode] + matrix_symbols[0] + tensor_symbols[mode + 1:]
    #print(tensor_symbols, matrix_symbols, result_symbols)

    result = torch.einsum(f"{tensor_symbols}, {matrix_symbols} -> {result_symbols}", tensor, matrix)
    return result



    