"""
PopulationIO: Handles population persistence (metadata, load, save, delete, clone).
"""
import os
import json
import glob
import re
import shutil
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any

from .serialization import _population_to_dict, _dict_to_genome, _node_to_dict

BASE_OUTPUT_DIR = "populations"


class PopulationIO:
    """Manages filesystem operations: paths, metadata, save/load/delete/clone."""

    def __init__(self, base_output_dir: str = BASE_OUTPUT_DIR):
        self.base_output_dir = base_output_dir

    def _set_paths(self, pop_name: str) -> Dict[str, str]:
        """Compute and return directory paths for a population."""
        safe_name = "".join(
            [c if c.isalnum() or c in (' ', '_', '-') else '_' for c in pop_name]
        ).strip()
        safe_name = safe_name.replace(' ', '_')

        base_dir = os.path.join(self.base_output_dir, safe_name)
        return {
            "root": base_dir,
            "saves": os.path.join(base_dir, "saves"),
            "genome_history": os.path.join(base_dir, "history"),
        }

    def _find_latest_gen_file(self, dirs: Dict[str, str]) -> Tuple[Optional[str], int]:
        """Find the latest generation JSON file for a population."""
        gen_files = glob.glob(os.path.join(dirs["genome_history"], "gen_*.json"))

        if not gen_files:
            return None, -1

        latest_file = ""
        latest_gen_num = -1

        for f in gen_files:
            basename = os.path.basename(f)
            match = re.match(r'gen_(\d+)\.json', basename)
            if match:
                gen_num = int(match.group(1))
                if gen_num > latest_gen_num:
                    latest_gen_num = gen_num
                    latest_file = f

        return latest_file, latest_gen_num

    def list_populations(self) -> List[Dict[str, Any]]:
        """List all populations with metadata."""
        metadata_list = []
        os.makedirs(self.base_output_dir, exist_ok=True)

        for pop_name in os.listdir(self.base_output_dir):
            pop_dir = os.path.join(self.base_output_dir, pop_name)
            if not os.path.isdir(pop_dir):
                continue

            metadata_file = os.path.join(pop_dir, "metadata.json")
            if not os.path.exists(metadata_file):
                continue

            try:
                dirs = self._set_paths(pop_name)

                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data_meta = json.load(f)
                config = data_meta.get('initial_config', {})

                latest_gen_file, latest_gen_num = self._find_latest_gen_file(dirs)

                mean_current_genes = {"primitives": 0.0, "operations": 0.0}
                if latest_gen_num > -1 and latest_gen_file:
                    with open(latest_gen_file, 'r', encoding='utf-8') as f:
                        data_latest_gen = json.load(f)

                    pop_data = data_latest_gen['data']['population']
                    genomes = [
                        _dict_to_genome(item['composition'])
                        for item in pop_data
                        if 'composition' in item
                    ]

                    if genomes:
                        try:
                            total_primitives = 0
                            total_operations = 0
                            sizes = [g.size() for g in genomes]

                            for size_dict in sizes:
                                total_primitives += size_dict.get("primitives", 0)
                                total_operations += size_dict.get("operations", 0)

                            mean_current_genes = {
                                "primitives": total_primitives / len(genomes),
                                "operations": total_operations / len(genomes)
                            }
                        except Exception as e:
                            print(f"Warning: Could not compute genome.size() for {pop_name}: {e}")
                            mean_current_genes = {"primitives": -1, "operations": -1}

                metadata = {
                    "name": pop_name,
                    "path": pop_dir,
                    "size": config.get('population_size', 'N/A'),
                    "initial_genes": config.get('initial_genes', 'N/A'),
                    "current_gen": latest_gen_num,
                    "mean_current_genes": mean_current_genes
                }
                metadata_list.append(metadata)

            except Exception as e:
                print(f"Error reading metadata for {pop_name}: {e}")

        return metadata_list

    def load_population_data(self, pop_name: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Load latest generation data and return (data, dirs)."""
        dirs = self._set_paths(pop_name)

        latest_json, latest_gen = self._find_latest_gen_file(dirs)

        if not latest_json:
            raise FileNotFoundError(f"No generation files found for {pop_name}")

        with open(latest_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data, dirs

    def load_generation(self, pop_name: str, generation: int) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Load a specific generation snapshot for a population."""
        dirs = self._set_paths(pop_name)
        os.makedirs(dirs["genome_history"], exist_ok=True)

        json_path = os.path.join(dirs["genome_history"], f"gen_{generation}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"gen_{generation}.json not found for {pop_name}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data, dirs

    def initialize_directories(self, pop_name: str) -> Dict[str, str]:
        """Create necessary directories for a new population."""
        dirs = self._set_paths(pop_name)
        os.makedirs(dirs["saves"], exist_ok=True)
        os.makedirs(dirs["genome_history"], exist_ok=True)
        return dirs

    def save_metadata(self, pop_name: str, config: Dict[str, Any]) -> None:
        """Save population metadata."""
        dirs = self._set_paths(pop_name)
        metadata_path = os.path.join(dirs["root"], "metadata.json")
        metadata_content = {
            "name": pop_name,
            "creation_date": datetime.now().isoformat(),
            "initial_config": config
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_content, f, indent=2)

    def save_generation(self, pop_name: str, generation: int, config: Dict[str, Any], population: List[Any]) -> None:
        """Save a generation snapshot."""
        dirs = self._set_paths(pop_name)
        json_filename = f"gen_{generation}.json"
        json_path = os.path.join(dirs["genome_history"], json_filename)

        meta = {
            "generation": generation,
            "config": config,
            "data": _population_to_dict(population),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def delete_population(self, pop_name: str) -> None:
        """Delete a population and all its files."""
        dirs = self._set_paths(pop_name)

        if not os.path.isdir(dirs["root"]):
            raise FileNotFoundError(f"Population directory does not exist: {dirs['root']}")

        shutil.rmtree(dirs["root"])

    def clone_population(self, original_name: str, new_name: str) -> None:
        """Clone a population."""
        src_dirs = self._set_paths(original_name)
        dst_dirs = self._set_paths(new_name)

        src_dir = src_dirs["root"]
        dst_dir = dst_dirs["root"]

        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Source directory not found: {src_dir}")

        if os.path.exists(dst_dir):
            raise FileExistsError(f"Destination population name already exists: {dst_dir}")

        shutil.copytree(src_dir, dst_dir)
        print(f"Successfully cloned to: {dst_dir}")
