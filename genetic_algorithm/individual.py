"""
GA individuals.

An individual is a candidate design plus the fitness it achieved. Designs are
drawn from, and mutated within, a :class:`~genetic_algorithm.design_space.DesignSpace`,
so the same individual type serves every application rather than hard-coding one
chromosome layout per application.

Individuals are ordered by fitness, so ``max(population)`` returns the fittest.
"""

from functools import total_ordering
from typing import Any, Dict, Optional
import copy
import numpy as np

from genetic_algorithm.design_space import DesignSpace


@total_ordering
class Individual:
    """A candidate design and its measured fitness."""

    def __init__(
        self,
        design_space: DesignSpace,
        design: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            design_space: Space the design is drawn from.
            design: An explicit design; a random one is sampled when omitted.
            rng: Random generator used for sampling.
        """
        self.design_space = design_space
        self.design = design if design is not None else design_space.sample(rng)
        self.fitness: float = float('-inf')
        self.evaluated = False

    # ------------------------------------------------------------------

    def evaluate(self, fitness_fn, timestep: int = 0) -> float:
        """Measure and cache this individual's fitness.

        Args:
            fitness_fn: Callable ``(design, timestep) -> float``.
            timestep: Timestep the evaluation belongs to.

        Returns:
            The fitness value.
        """
        self.fitness = float(fitness_fn(self.design, timestep))
        self.evaluated = True
        return self.fitness

    def copy(self) -> 'Individual':
        """Deep-copy this individual, fitness included."""
        clone = Individual(self.design_space, copy.deepcopy(self.design))
        clone.fitness = self.fitness
        clone.evaluated = self.evaluated
        return clone

    def encode(self) -> np.ndarray:
        """Fixed-length numeric encoding of this individual's design."""
        return self.design_space.encode(self.design)

    # ------------------------------------------------------------------
    # Ordering by fitness
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness == other.fitness

    def __lt__(self, other) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.4f}, evaluated={self.evaluated})"
