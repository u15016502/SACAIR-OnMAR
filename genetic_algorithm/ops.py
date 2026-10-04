"""
Genetic operators: selection, crossover, mutation.

The operators work on :class:`~genetic_algorithm.individual.Individual` objects
and delegate the design-level work to the individual's
:class:`~genetic_algorithm.design_space.DesignSpace`, so they are independent of
which application is being optimised.
"""

from typing import List, Optional, Tuple
import numpy as np

from genetic_algorithm.individual import Individual


def _as_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def select(
    population: List[Individual],
    tournament_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Individual:
    """Tournament selection.

    Draws ``tournament_size`` individuals at random and returns the fittest.
    Selection pressure rises with the tournament size.

    Args:
        population: Population to select from.
        tournament_size: Number of competitors per tournament.
        rng: Random generator.

    Returns:
        The winning individual (not copied - callers that mutate it must copy).
    """
    if not population:
        raise ValueError("Cannot select from an empty population")

    rng = _as_rng(rng)
    size = min(tournament_size, len(population))
    contender_indices = rng.choice(len(population), size=size, replace=False)
    contenders = [population[int(i)] for i in contender_indices]
    return max(contenders)


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Individual, Individual]:
    """Recombine two parents into two offspring.

    Args:
        parent_a: First parent.
        parent_b: Second parent.
        rng: Random generator.

    Returns:
        Two new, unevaluated offspring.
    """
    rng = _as_rng(rng)
    space = parent_a.design_space

    child_a = Individual(space, space.crossover(parent_a.design, parent_b.design, rng))
    child_b = Individual(space, space.crossover(parent_b.design, parent_a.design, rng))
    return child_a, child_b


def mutation(
    individual: Individual,
    gene_rate: float = 0.1,
    structure_rate: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Individual:
    """Mutate an individual.

    Args:
        individual: Individual to mutate.
        gene_rate: Per-gene resampling probability.
        structure_rate: Probability of adding or removing a block (layer).
        rng: Random generator.

    Returns:
        A new, unevaluated individual.
    """
    rng = _as_rng(rng)
    space = individual.design_space
    mutated_design = space.mutate(
        individual.design, rng, gene_rate=gene_rate, structure_rate=structure_rate
    )
    return Individual(space, mutated_design)
