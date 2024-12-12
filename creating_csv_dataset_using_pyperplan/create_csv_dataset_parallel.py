import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pyperplan.planner import (
    HEURISTICS,
    SEARCHES,
    search_plan,
    validate_solution,
)
from get_params_for_run import get_all_dirs_in_dir, get_all_domain_problem_pairs

# possible search algorithms: astar, wastar, gbf, bfs, ehs, ids, sat
# possible heuristics: landmark, lmcut, hadd, hff, hmax, hsa, blind

SEARCH_ALGORITHMS_LIST = ['astar', 'gbf']
HEURISTICS_LIST = ['lmcut', 'hff']

def setup_logger():
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Include a timestamp in the log filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"run_{timestamp_str}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Log level for the logger

    # File handler for logging debug messages to a file
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=5_000_000, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Log all messages, including DEBUG

    # Console handler for logging messages to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only log INFO and above to the terminal

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def run_search_task(domain_file, problem_file, search_algorithm, heuristic, validate, output_path, logger):
    try:
        logger.debug(f"Starting search task: domain={domain_file}, problem={problem_file}, "
                     f"search_algorithm={search_algorithm}, heuristic={heuristic}")

        # Find domain and problem files
        domain_file = os.path.abspath(domain_file)
        problem_file = os.path.abspath(problem_file)

        logger.debug("Resolved file paths for domain and problem.")

        # Run the search algorithm
        heuristic_fn = HEURISTICS[heuristic]
        search_fn = SEARCHES[search_algorithm]

        logger.debug("Running search_plan...")
        plan = search_plan(domain_file, problem_file, search_fn, heuristic_fn, output_path)

        if validate:
            logger.debug("Validating solution...")
            validate_solution(domain_file, problem_file, plan)
            logger.debug("Validation completed successfully.")

        logger.info(f"Completed: {search_algorithm}, {heuristic}, {domain_file}, {problem_file}")
        return output_path
    except Exception as e:
        logger.error(f"Error in task {search_algorithm}, {heuristic}, {domain_file}, {problem_file}: {e}", exc_info=True)
        return None

def main():
    logger = setup_logger()
    logger.info("Starting the planning run...")

    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "dataset"
    logger.debug(f"Dataset directory set to {dataset_dir}")

    dataset_subdirs = get_all_dirs_in_dir(dataset_dir)
    logger.debug("Retrieved dataset subdirectories.")

    search_parameters_list = []
    for dataset_subdir in dataset_subdirs:
        logger.debug(f"Processing dataset subdir: {dataset_subdir}")
        domain_problem_pairs = get_all_domain_problem_pairs(dataset_subdir)
        for domain_file, problem_file in domain_problem_pairs:
            domain_name = dataset_subdir.name
            task_number = problem_file.stem.split("task")[1]
            for search_algorithm in SEARCH_ALGORITHMS_LIST:
                for heuristic in HEURISTICS_LIST:
                    output_path = root_dir / "csv_output" / f"{search_algorithm}_{heuristic}_{domain_name}_{task_number}.csv"
                    if not output_path.parent.exists():
                        output_path.parent.mkdir(parents=True)
                        logger.debug(f"Created output directory {output_path.parent}")
                    search_parameters_list.append((domain_file, problem_file, search_algorithm, heuristic, True, output_path))

    logger.info(f"Number of search parameters: {len(search_parameters_list)}")

    max_workers = 32  # Number of cores to use
    logger.info(f"Using {max_workers} cores for parallel processing.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(run_search_task, *params, logger): params for params in search_parameters_list
        }

        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"Task completed successfully: {params}")
                else:
                    logger.warning(f"Task completed with no result: {params}")
            except Exception as exc:
                logger.error(f"Task failed: {params} with exception {exc}", exc_info=True)

    logger.info("All tasks have completed.")

if __name__ == "__main__":
    main()
