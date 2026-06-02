"""
Minimal CodeFormer architecture package.

This package provides the CodeFormer network architecture without the full repository.
Face detection and restoration helpers are provided by facexlib.

Usage:
    from codeformer_minimal import codeformer_arch  # noqa: F401
    from basicsr.utils.registry import ARCH_REGISTRY
    CodeFormer = ARCH_REGISTRY.get('CodeFormer')
"""

__all__ = ['codeformer_arch', 'vqgan_arch']
