# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Grid ("checkerboard") artifacts that perceptual training adds on the block grid of the decoder's pixel shuffle."""

import torch


def _grid_mask(tile: int, period: int, device) -> torch.Tensor:
    step = tile // period
    on = torch.arange(tile, device=device) % step == 0
    grid = on[:, None] & on[None, :]
    grid[0, 0] = False
    return grid


def _tile_power(x: torch.Tensor, tile: int) -> torch.Tensor:
    """Mean power spectrum over origin-aligned tiles: (B, 1, H, W) -> (B, tile, tile)."""
    h, w = x.shape[-2] // tile * tile, x.shape[-1] // tile * tile
    if h == 0 or w == 0:
        raise ValueError(f"image {tuple(x.shape[-2:])} is smaller than the {tile}x{tile} tile")
    tiles = x[..., :h, :w].float().unfold(-2, tile, tile).unfold(-2, tile, tile)
    tiles = tiles - tiles.mean(dim=(-1, -2), keepdim=True)
    return torch.fft.fft2(tiles).abs().square().flatten(1, -3).mean(dim=1)


def grid_ratio_db(x: torch.Tensor, *, period: int = 8, tile: int = 64, offset: int = 2) -> torch.Tensor:
    """Per image, mean power at the grid harmonics relative to nearby bins in dB; about 0 without a grid."""
    if tile % period:
        raise ValueError(f"tile {tile} must be a multiple of period {period}")
    power = _tile_power(x, tile)
    eps = torch.finfo(power.dtype).tiny
    neighbors = torch.stack([torch.roll(power, shifts=s, dims=d) for s in (offset, -offset) for d in (-2, -1)]).mean(
        dim=0
    )
    grid = _grid_mask(tile, period, x.device)
    ratio_db = 10 * (torch.log10(power[:, grid].clamp_min(eps)) - torch.log10(neighbors[:, grid].clamp_min(eps)))
    return ratio_db.mean(dim=1)


def grid_pattern_energy(x: torch.Tensor, *, period: int = 8, tile: int = 64) -> torch.Tensor:
    """Mean square of the grid pattern per image and channel, (B, C, H, W) -> (B, C): the power at the grid harmonics.

    Each tile is averaged onto one ``period x period`` block, whose mean is removed.
    """
    if tile % period:
        raise ValueError(f"tile {tile} must be a multiple of period {period}")
    h, w = x.shape[-2] // tile * tile, x.shape[-1] // tile * tile
    if h == 0 or w == 0:
        raise ValueError(f"image {tuple(x.shape[-2:])} is smaller than the {tile}x{tile} tile")
    b, c = x.shape[:2]
    n = tile // period
    blocks = x[..., :h, :w].reshape(b, c, h // tile, n, period, w // tile, n, period)
    pattern = blocks.mean(dim=(3, 6))
    pattern = pattern - pattern.mean(dim=(3, 5), keepdim=True)
    return pattern.square().mean(dim=(2, 3, 4, 5))
