"""
PopulationStateManager: Manages in-memory population, evolution, and mutation config.
"""
import numpy as np
from typing import List, Sequence, Set, Dict, Any
from dataclasses import replace

try:
    from src.core.evolution import GAConfig, initialize_population, evolve_one_generation, CompositionGenome
except ImportError:                                                 
    from evolution import GAConfig, initialize_population, evolve_one_generation, CompositionGenome                        

try:
    from src.core.evolution.genome import MutationConfig
except ImportError:                                                 
    from evolution.genome import MutationConfig                        


class PopulationStateManager:
    """Manages in-memory population state, RNG, and evolution."""

    def __init__(self):
        self.population: List[CompositionGenome] = []
        self.cfg: GAConfig | None = None
        self.live_mutation_cfg: MutationConfig | None = None
        self.rng: np.random.Generator | None = None

    def initialize(self, pop_size: int, genes: int, seed: int | None) -> List[CompositionGenome]:
        """Initialize population with given parameters."""
        self.cfg = GAConfig(
            population_size=pop_size,
            num_genes=genes,
            random_seed=seed if seed is not None else np.random.randint(0, 1_000_000)
        )
        self.live_mutation_cfg = self.cfg.mutation
        self.rng, self.population = initialize_population(self.cfg)
        return self.population

    def load_from_data(self, config_data: Dict[str, Any], pop_data: List[Dict[str, Any]]) -> None:
        """Load population from serialized data."""
        self.cfg = GAConfig(
            population_size=config_data['population_size'],
            num_genes=config_data['num_genes'],
            random_seed=config_data['random_seed']
        )
        self.live_mutation_cfg = self.cfg.mutation
        self.rng = np.random.default_rng(self.cfg.random_seed)

        from .serialization import _dict_to_genome
        self.population = []
        for item in pop_data:
            comp = item.get("composition")
            if comp:
                genome = _dict_to_genome(comp)
                self.population.append(genome)

    def evolve(self, selected_indices: Sequence[int]) -> List[CompositionGenome]:
        """Evolve population to next generation."""
        if not selected_indices:
            return self.population

        if self.cfg is None:
            return self.population

        if self.rng is None:
            self.rng = np.random.default_rng(self.cfg.random_seed)

        if self.live_mutation_cfg is None:
            self.live_mutation_cfg = self.cfg.mutation

        likes_list = list(selected_indices)
        temp_cfg = replace(self.cfg, mutation=self.live_mutation_cfg)

        self.population = evolve_one_generation(
            self.rng,
            self.population,
            likes_list,
            temp_cfg
        )
        return self.population

    def get_mutation_config(self) -> MutationConfig:
        """Get current mutation configuration."""
        if self.live_mutation_cfg is None:
            return MutationConfig()
        return self.live_mutation_cfg

    def update_mutation_config(self, param_name: str, value: float) -> None:
        """Update a mutation parameter."""
        if self.live_mutation_cfg is None:
            return

        try:
            new_cfg = replace(self.live_mutation_cfg, **{param_name: value})
            self.live_mutation_cfg = new_cfg
        except Exception as e:
            print(f"Error updating mutation parameter {param_name}: {e}")

    def get_average_genome_size(self) -> Dict[str, float]:
        """Compute average size (primitives and operations) across population."""
        if not self.population:
            return {"primitives": 0.0, "operations": 0.0}
        try:
            total_primitives = 0
            total_operations = 0
            sizes = [genome.size() for genome in self.population]
            for size_dict in sizes:
                total_primitives += size_dict.get("primitives", 0)
                total_operations += size_dict.get("operations", 0)

            count = len(self.population)
            if count == 0:
                return {"primitives": 0.0, "operations": 0.0}

            return {
                "primitives": total_primitives / count,
                "operations": total_operations / count
            }
        except Exception as e:
            print(f"Error calculating genome size: {e}")
            return {"primitives": -1.0, "operations": -1.0}

    def get_individual_genome_size(self, index: int) -> Dict[str, int]:
        """Get size of an individual genome."""
        if not (0 <= index < len(self.population)):
            return {"primitives": -1, "operations": -1}
        try:
            return self.population[index].size()
        except Exception as e:
            print(f"Error in get_individual_genome_size (index {index}): {e}")
            return {"primitives": -1, "operations": -1}

    def get_individual_composition_dict(self, index: int) -> Dict[str, Any] | None:
        """Get composition dict for an individual."""
        if 0 <= index < len(self.population):
            from .serialization import _node_to_dict
            genome = self.population[index]
            return _node_to_dict(genome.root)
        return None
