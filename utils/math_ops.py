# utils/math_ops.py
import torch

def symmetric(A):
    """Return the symmetric part of a tensor."""
    size = list(range(len(A.shape)))
    temp = size[-1]
    size.pop()
    size.insert(-1, temp)
    return 0.5 * (A + A.permute(*size))

def matrix_operator(A, operator):
    """Applies a matrix operator such as sqrtm, rsqrtm, logm, or expm."""
    try:
        u, s, v = A.svd()
    except RuntimeError as e:
        print("SVD failed:", e)
        identity = torch.eye(A.shape[-1], device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
        return identity

    if operator == 'sqrtm':
        s = s.sqrt()
    elif operator == 'rsqrtm':
        s = s.rsqrt()
    elif operator == 'logm':
        s = torch.clamp(s, min=1e-6).log()
    elif operator == 'expm':
        s = s.exp()
    else:
        raise ValueError(f'Operator {operator} is not implemented')

    return u @ torch.diag_embed(s) @ v.transpose(-1, -2)

def tangent_space(A, ref, inverse_transform=False):
    """Transforms A to/from the tangent space using the reference matrix."""
    ref_sqrt = matrix_operator(ref, 'sqrtm')
    ref_sqrt_inv = matrix_operator(ref, 'rsqrtm')
    middle = ref_sqrt_inv @ A @ ref_sqrt_inv
    middle = matrix_operator(middle, 'logm' if inverse_transform else 'expm')
    return ref_sqrt @ middle @ ref_sqrt

def untangent_space(A, ref):
    """Inverse of the tangent space mapping."""
    return tangent_space(A, ref, inverse_transform=True)

def parallel_transform(A, ref1, ref2):
    """Parallel transport from ref1 to ref2."""
    out = untangent_space(A, ref1)
    return tangent_space(out, ref2)
