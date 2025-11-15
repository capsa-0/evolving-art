"""BackendAdapter: Facade composing PopulationIO, PopulationStateManager, and rendering helpers."""
import os

from src.rendering import genome_to_png_bytes, save_genome_as_png, save_genome_as_svg

from .population_io import PopulationIO, BASE_OUTPUT_DIR
from .population_state_manager import PopulationStateManager

GUI_IMAGE_RES = 250
GUI_HQ_RES = 1200


class BackendAdapter:
    """
    Thin facade composing three focused modules:
    - PopulationIO: filesystem and metadata
    - PopulationStateManager: in-memory population and evolution
    - Vectorizer helpers: direct rendering to PNG/SVG via matplotlib layer
    """

    def __init__(self):
        self.pop_io = PopulationIO(BASE_OUTPUT_DIR)
        self.pop_state = PopulationStateManager()
        self.pop_name = ""
        self.generation = 0
        self.dirs = {}

    @property
    def population(self):
        """Expose population for compatibility."""
        return self.pop_state.population

    @property
    def cfg(self):
        """Expose cfg for compatibility."""
        return self.pop_state.cfg

    @property
    def live_mutation_cfg(self):
        """Expose mutation config for compatibility."""
        return self.pop_state.live_mutation_cfg

    def list_populations(self):
        """List all populations."""
        return self.pop_io.list_populations()

    def load_population(self, pop_name):
        """Load a population from disk."""
        data, dirs = self.pop_io.load_population_data(pop_name)
        config_data = data['config']
        pop_data = data['data']['population']
        self.pop_state.load_from_data(config_data, pop_data)
        self.pop_name = pop_name
        self.generation = data.get('generation', 0)
        self.dirs = dirs
        return {"pop_name": pop_name, "generation": self.generation}

    def initialize(self, pop_size, genes, seed, pop_name="Simulation"):
        """Initialize a new population."""
        dirs = self.pop_io.initialize_directories(pop_name)
        config = {
            "population_size": pop_size,
            "initial_genes": genes,
            "seed": seed
        }
        self.pop_io.save_metadata(pop_name, config)
        self.pop_state.initialize(pop_size, genes, seed)
        self.pop_name = pop_name
        self.generation = 0
        self.dirs = dirs
                                                                                             
        self.save_generation_state(self.generation)
        return self.pop_state.population

    def evolve(self, selected_indices):
        """Evolve population to next generation."""
        return self.pop_state.evolve(selected_indices)

    def render_all_previews_parallel(self):
        """Render all population individuals in parallel and return PNG bytes."""
        return [
            genome_to_png_bytes(genome, resolution=GUI_IMAGE_RES)
            for genome in self.pop_state.population
        ]

    def save_hq_individual(self, index, generation, fmt="png"):
        """Save an individual in high quality."""
        if index >= len(self.pop_state.population):
            raise IndexError(f"Individual index {index} out of range")
        folder = self.dirs["saves"]
        os.makedirs(folder, exist_ok=True)
        filename = f"gen_{generation}_ind_{index}.{fmt}"
        full_path = os.path.join(folder, filename)
        genome = self.pop_state.population[index]
        fmt_lower = fmt.lower()
        if fmt_lower == "png":
            save_genome_as_png(genome, filename=full_path, resolution=GUI_HQ_RES)
        elif fmt_lower == "svg":
            save_genome_as_svg(genome, filename=full_path)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'png' or 'svg'.")
        return full_path

    def save_generation_state(self, generation):
        """Save current generation state to disk."""
        if not self.dirs:
            raise RuntimeError("Population not initialized. Call initialize() or load_population() first.")
        config = {
            "population_size": self.pop_state.cfg.population_size if self.pop_state.cfg else 0,
            "num_genes": self.pop_state.cfg.num_genes if self.pop_state.cfg else 0,
            "random_seed": self.pop_state.cfg.random_seed if self.pop_state.cfg else 0,
        }
        self.pop_io.save_generation(
            self.pop_name,
            generation,
            config,
            self.pop_state.population
        )

    def load_generation(self, generation: int):
        """Load a specific generation snapshot into memory."""
        if not self.pop_name:
            raise RuntimeError("No population loaded. Call load_population() first.")

        data, dirs = self.pop_io.load_generation(self.pop_name, generation)

        config_data = data.get("config")
        population_data = data.get("data", {}).get("population", [])
        if not config_data:
            raise KeyError(f"Generation file does not contain valid configuration for gen {generation}.")

        self.pop_state.load_from_data(config_data, population_data)
        self.generation = data.get("generation", generation)
        self.dirs = dirs
        return self.pop_state.population

    def get_average_genome_size(self):
        """Get average genome size."""
        return self.pop_state.get_average_genome_size()

    def get_individual_genome_size(self, index):
        """Get size of an individual genome."""
        return self.pop_state.get_individual_genome_size(index)

    def get_individual_composition_dict(self, index):
        """Get composition dict for an individual."""
        return self.pop_state.get_individual_composition_dict(index)

    def delete_population(self, pop_name):
        """Delete a population."""
        self.pop_io.delete_population(pop_name)

    def clone_population(self, original_name, new_name):
        """Clone a population."""
        self.pop_io.clone_population(original_name, new_name)

    def get_mutation_config(self):
        """Get mutation config."""
        return self.pop_state.get_mutation_config()

    def update_mutation_config(self, param_name, value):
        """Update mutation config parameter."""
        self.pop_state.update_mutation_config(param_name, value)
