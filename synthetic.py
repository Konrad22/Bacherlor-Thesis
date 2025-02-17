import itertools
from enum import Enum

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from symbols import SymbolGenerator

Shape = tuple[int, ...]
Scalar = torch.float64


BALL_RADIUS = lambda shape: torch.min(torch.tensor(shape)).data // 3
CUBOID_SHAPE = lambda shape: torch.tensor(shape) // 2
DIAMOND_RADIUS = lambda shape: torch.min(torch.tensor(shape)).data // 3
SWISS_BORDER_CUBOID_SHAPE = lambda shape: (4 * torch.tensor(shape)) // 5
SWISS_SLIM_CUBOID_SHAPE = lambda shape: SWISS_BORDER_CUBOID_SHAPE(shape) // 3


class Pattern(Enum):
    Ball = "ball"
    Cuboid = "cuboid"
    Diamond = "diamond"
    Checkers = "checkers"
    Swiss = "swiss"


def outer_product(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    symbol_generator = SymbolGenerator()
    symbols_1 = symbol_generator.generate(len(tensor1.shape))
    symbols_2 = symbol_generator.generate(len(tensor2.shape))
    return torch.einsum(f"{symbols_1}, {symbols_2} -> {symbols_1}{symbols_2}", tensor1, tensor2)


def gen_synthetic(pattern: Pattern, shape: Shape, dtype=Scalar) -> Tensor:
    synthetic_tensor: Tensor = torch.zeros(shape, dtype=dtype)
    # create a grid of indices of shape (len(shape), *shape) where the first axis is the dimension which is counted along
    index_grid = torch.stack(torch.meshgrid(*[torch.arange(x) for x in shape], indexing='ij'))
    middle = torch.tensor(shape) // 2
    broadcast_middle = outer_product(middle, torch.ones(shape))
    if pattern == Pattern.Ball:
        # compute the distance from the middle
        radius = BALL_RADIUS(shape)
        condition = torch.sum((index_grid - broadcast_middle) ** 2, axis=0) <= radius ** 2
        synthetic_tensor[condition] = 1
        return synthetic_tensor
    if pattern == Pattern.Cuboid:
        broadcast_cuboid_shape = outer_product(CUBOID_SHAPE(shape), torch.ones(shape))
        condition = torch.all(torch.abs(index_grid - broadcast_middle) <= broadcast_cuboid_shape // 2, axis=0)
        synthetic_tensor[condition] = 1
        return synthetic_tensor
    if pattern == Pattern.Diamond:
        radius = DIAMOND_RADIUS(shape)
        condition = torch.sum(torch.abs(index_grid - broadcast_middle), axis=0) <= radius
        synthetic_tensor[condition] = 1
        return synthetic_tensor
    if pattern == Pattern.Checkers:
        for region in itertools.product([0, 1], repeat=len(shape)):
            start = torch.tensor([0 if axis_region == 0 else axis_size // 2 for axis_region, axis_size in zip(region, shape)])
            end = torch.tensor([axis_size // 2 if axis_region == 0 else axis_size for axis_region, axis_size in zip(region, shape)])
            start_broadcast = outer_product(start, torch.ones(shape))
            end_broadcast = outer_product(end, torch.ones(shape))
            condition = torch.all((start_broadcast <= index_grid) & (index_grid < end_broadcast), axis=0)
            synthetic_tensor[condition] = torch.sum(torch.tensor(region)) % 2
        return synthetic_tensor
    if pattern == Pattern.Swiss:
        # create a stretched cuboid for every axis and fill it with ones
        border_cuboid_shape = SWISS_BORDER_CUBOID_SHAPE(shape)
        slim_cuboid_shape = SWISS_SLIM_CUBOID_SHAPE(shape)
        for axis in range(len(shape)):
            cuboid_shape = torch.clone(slim_cuboid_shape)
            cuboid_shape[axis] = border_cuboid_shape[axis]
            broadcast_cuboid_shape = outer_product(cuboid_shape, torch.ones(shape))
            condition = torch.all(torch.abs(index_grid - broadcast_middle) <= broadcast_cuboid_shape // 2, axis=0)
            synthetic_tensor[condition] = 1
        return synthetic_tensor
    raise ValueError(f"Unknown pattern: {pattern}")


def main():
    shape_2d = (25, 25)
    shape_3d = (25, 25, 35)
    for pattern in Pattern:
        tensor = gen_synthetic(pattern, shape_2d)
        plt.imshow(tensor, cmap=plt.cm.OrRd, interpolation='nearest')
        plt.title(f"{pattern.value} pattern for shape {shape_2d}")
        plt.savefig(f"patterns/torch_{pattern.value}_2d.png")
        plt.clf()
        tensor = gen_synthetic(pattern, shape_3d)
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(tensor)
        plt.title(f"{pattern.value} pattern for shape {shape_3d}")
        plt.savefig(f"patterns/torch_{pattern.value}_3d.png")
        plt.clf()


if __name__ == "__main__":
    main()
