"""
The genetic algorithm used as the design algorithm for OnMAR/OffMAR.

In Algorithm 9 of Chapter 8, the design algorithm is called once per timestep
(``c <- design_algorithm(dataset, t)``) and returns a single design, which the
application then executes (``p <- aa.exec(c, dataset, t)``). This class
implements that contract: :meth:`step` advances the GA by one generation and
returns the fittest design found so far.

The population *persists between timesteps*, which is what makes the search
cumulative rather than a fresh random draw each timestep. Carried-over elites
are re-evaluated at the new timestep by default, because fitness measured at an
earlier timestep is stale once the data or the training state has moved on.

Fitness evaluation is supplied by the caller as ``(design, timestep) -> float``;
for the CNN application it is a short proxy training run, so that a generation
costs a small multiple of one timestep rather than a full training run per
individual.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np

from genetic_algorithm.design_space import DesignSpace
from genetic_algorithm.individual import Individual
from genetic_algorithm import ops


class GeneticAlgorithm:
    """A steady-state GA that advances one generation per timestep."""

    def __init__(
        self,
        design_space: DesignSpace,
        fitness_fn: Callable[[Dict[str, Any], int], float],
        population_fitness_fn: Optional[Callable[[List[Dict[str, Any]], int], List[float]]] = None,
        population_size: int = 8,
        tournament_size: int = 3,
        crossover_rate: float = 0.75,
        mutation_rate: float = 0.25,
        gene_mutation_rate: float = 0.1,
        structure_mutation_rate: float = 0.15,
        elite_size: int = 1,
        generations_per_step: int = 1,
        reevaluate_elites: bool = True,
        random_seed: int = 42,
    ):
        """
        Args:
            design_space: Space to search.
            fitness_fn: Callable ``(design, timestep) -> float``.
            population_fitness_fn: Optional callable
                ``(designs, timestep) -> [float]`` scoring a whole generation
                at once. Supplied by applications that can evaluate
                candidates in parallel, which is where most of a run's time
                goes; without it candidates are scored one at a time.
            population_size: Individuals per generation.
            tournament_size: Competitors per selection tournament.
            crossover_rate: Probability that a pair is recombined.
            mutation_rate: Probability that an offspring is mutated.
            gene_mutation_rate: Per-gene resampling probability when mutating.
            structure_mutation_rate: Probability of adding/removing a layer.
            elite_size: Fittest individuals copied unchanged into the next
                generation.
            generations_per_step: Generations run per :meth:`step` call.
            reevaluate_elites: Re-measure carried-over elites at the new
                timestep instead of trusting a stale fitness.
            random_seed: Seed for the GA's own generator.
        """
        self.design_space = design_space
        self.fitness_fn = fitness_fn
        self.population_fitness_fn = population_fitness_fn
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.gene_mutation_rate = gene_mutation_rate
        self.structure_mutation_rate = structure_mutation_rate
        # At least one elite, but never the whole population: if elitism filled
        # every slot there would be no offspring and the search would freeze.
        self.elite_size = max(1, min(elite_size, max(1, population_size - 1)))
        self.generations_per_step = max(1, generations_per_step)
        self.reevaluate_elites = reevaluate_elites
        self.rng = np.random.default_rng(random_seed)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None

        # Cost accounting: fitness evaluations dominate GA runtime.
        self.num_fitness_evaluations = 0
        self.num_generations = 0

    # ------------------------------------------------------------------

    def initialise(self, timestep: int = 0, seed_design: Optional[Dict[str, Any]] = None) -> None:
        """Create and evaluate the initial population.

        Args:
            timestep: Timestep to evaluate the initial population at.
            seed_design: Optional design to inject, so a known-good incumbent
                is not lost at startup.
        """
        self.population = []

        if seed_design is not None and self.design_space.validate(seed_design):
            self.population.append(Individual(self.design_space, seed_design))

        while len(self.population) < self.population_size:
            self.population.append(Individual(self.design_space, rng=self.rng))

        self._evaluate(self.population, timestep)
        self._track_best()

    def step(self, timestep: int, seed_design: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Advance the GA and return the best design found so far.

        This is the ``design_algorithm(dataset, t)`` of Algorithm 9.

        Args:
            timestep: Current timestep, passed through to the fitness function.
            seed_design: Design to inject when initialising on the first call.

        Returns:
            The fittest design known to the GA.
        """
        if not self.population:
            self.initialise(timestep, seed_design)
            self.num_generations += 1
            return self._best_design()

        for _ in range(self.generations_per_step):
            self._run_generation(timestep)
            self.num_generations += 1

        return self._best_design()

    # ------------------------------------------------------------------

    def _run_generation(self, timestep: int) -> None:
        """Produce and evaluate one new generation."""
        # Elitism: the fittest survive unchanged.
        ranked = sorted(self.population, reverse=True)
        elites = [individual.copy() for individual in ranked[:self.elite_size]]

        # A stale fitness from an earlier timestep would let a design that was
        # good then dominate the population now, so elites are re-measured -
        # but in the same batch as the offspring below, so a parallel
        # evaluator sees the whole generation at once instead of one small
        # call followed by a larger one.
        if self.reevaluate_elites:
            for elite in elites:
                elite.evaluated = False

        offspring: List[Individual] = []
        while len(offspring) < self.population_size - len(elites):
            parent_a = ops.select(self.population, self.tournament_size, self.rng)
            parent_b = ops.select(self.population, self.tournament_size, self.rng)

            if self.rng.random() < self.crossover_rate:
                child_a, child_b = ops.crossover(parent_a, parent_b, self.rng)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
                child_a.evaluated = child_b.evaluated = False

            for child in (child_a, child_b):
                if self.rng.random() < self.mutation_rate:
                    child = ops.mutation(
                        child,
                        gene_rate=self.gene_mutation_rate,
                        structure_rate=self.structure_mutation_rate,
                        rng=self.rng,
                    )
                if len(offspring) < self.population_size - len(elites):
                    offspring.append(child)

        self._evaluate(elites + offspring, timestep)
        self.population = elites + offspring
        self._track_best()

    def _evaluate(self, individuals: List[Individual], timestep: int, force: bool = False) -> None:
        """Evaluate individuals, skipping those already measured.

        Everything still needing a score is submitted together, so an
        application that evaluates candidates in parallel sees the whole batch
        at once rather than one design at a time.
        """
        pending = [i for i in individuals if force or not i.evaluated]
        if not pending:
            return

        if self.population_fitness_fn is not None:
            fitnesses = self.population_fitness_fn([i.design for i in pending], timestep)
            for individual, fitness in zip(pending, fitnesses):
                individual.fitness = float(fitness)
                individual.evaluated = True
                self.num_fitness_evaluations += 1
            return

        for individual in pending:
            individual.evaluate(self.fitness_fn, timestep)
            self.num_fitness_evaluations += 1

    def _track_best(self) -> None:
        """Keep the best-ever individual."""
        if not self.population:
            return
        generation_best = max(self.population)
        if self.best_individual is None or generation_best.fitness > self.best_individual.fitness:
            self.best_individual = generation_best.copy()

    def _best_design(self) -> Dict[str, Any]:
        """The best design known, falling back to a random sample."""
        if self.best_individual is not None:
            return self.best_individual.design
        if self.population:
            return max(self.population).design
        return self.design_space.sample(self.rng)

    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Summary of GA effort and current population state."""
        fitnesses = [i.fitness for i in self.population if i.evaluated]
        return {
            'num_generations': self.num_generations,
            'num_fitness_evaluations': self.num_fitness_evaluations,
            'population_size': len(self.population),
            'best_fitness': self.best_individual.fitness if self.best_individual else None,
            'mean_fitness': float(np.mean(fitnesses)) if fitnesses else None,
            'fitness_std': float(np.std(fitnesses)) if fitnesses else None,
        }
