"""
model package

Provides tools for preparing datasets for training and evaluation.

Modules:
    ground_truth: Functions for generating ground truth.

Exports:
    generate_ground_truth

Last Update:
    Owner: Kartik M. Jalal
    Date: 12/08/2025
"""

from .ground_truth import generate_ground_truth

__all__ = [
    "generate_ground_truth"
]