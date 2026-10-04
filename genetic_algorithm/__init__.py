"""
Genetic algorithm used as the design algorithm for OnMAR and OffMAR.

    DesignSpace       - declares what a design may be; samples, mutates,
                        recombines, encodes and decodes designs
    Individual        - a candidate design plus its measured fitness
    ops               - selection, crossover and mutation operators
    GeneticAlgorithm  - the driver; one generation per timestep
"""

from genetic_algorithm.design_space import DesignSpace, space_from_legacy
from genetic_algorithm.individual import Individual
from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm import ops

__all__ = [
    'DesignSpace',
    'space_from_legacy',
    'Individual',
    'GeneticAlgorithm',
    'ops',
]
