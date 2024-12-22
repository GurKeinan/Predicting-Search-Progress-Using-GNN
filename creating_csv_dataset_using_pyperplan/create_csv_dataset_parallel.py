import os
import signal
import psutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from get_params_for_run import get_all_dirs_in_dir, get_all_domain_problem_pairs
from pyperplan.planner import HEURISTICS, SEARCHES, search_plan, validate_solution

# Timeout duration in seconds (12 hours)
TASK_TIMEOUT = 12 * 60 * 60

def setup_logger():
    """Set up a logger focusing on essential events."""
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"run_{timestamp_str}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(log_filename, maxBytes=5_000_000, backupCount=5)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Remove any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def validate_task_inputs(domain_file, problem_file, logger):
    """Validate the existence of required input files."""
    if not os.path.isfile(domain_file):
        logger.warning(f"Domain file not found: {domain_file}")
        return False
    if not os.path.isfile(problem_file):
        logger.warning(f"Problem file not found: {problem_file}")
        return False
    return True

def terminate_process_tree(pid, logger):
    """Terminate a process and its child processes."""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        gone, alive = psutil.wait_procs([parent] + parent.children(), timeout=5)
        if alive:
            for proc in alive:
                proc.kill()
        logger.info(f"Terminated process tree for PID {pid}.")
    except psutil.NoSuchProcess:
        logger.warning(f"Process with PID {pid} does not exist.")
    except Exception as e:
        logger.error(f"Error terminating process tree for PID {pid}: {e}")

def run_search_task(domain_file, problem_file, search_algorithm, heuristic, validate, output_path, logger):
    """Perform the search task."""
    try:
        domain_name = Path(domain_file).parent.name
        problem_name = Path(problem_file).stem
        logger.info(f"Running task: domain={domain_name}, problem={problem_name}, "
                    f"search={search_algorithm}, heuristic={heuristic}")
        heuristic_fn = HEURISTICS[heuristic]
        search_fn = SEARCHES[search_algorithm]

        plan = search_plan(domain_file, problem_file, search_fn, heuristic_fn, output_path)
        if validate:
            validate_solution(domain_file, problem_file, plan)
        logger.info(f"Task completed: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error during task (domain={domain_file}, problem={problem_file}): {e}", exc_info=True)
        return None

def run_search_task_with_timeout(params, logger):
    """Wrapper for task execution with an enforced timeout."""
    pid = os.getpid()

    def alarm_handler(signum, frame):
        raise TimeoutError("Task timed out")

    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(TASK_TIMEOUT)

    try:
        result = run_search_task(*params, logger)
        signal.alarm(0)  # Cancel the alarm if the task completed in time
        return result
    except TimeoutError:
        logger.error(f"Task exceeded the timeout: {params}")
        terminate_process_tree(pid, logger)
        return None
    except Exception as e:
        logger.error(f"Task error (PID {pid}): {e}", exc_info=True)
        return None
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)

def prepare_tasks(dataset_dir, root_dir, logger):
    """Prepare the list of tasks to execute."""
    dataset_subdirs = get_all_dirs_in_dir(dataset_dir)
    logger.info(f"Found {len(dataset_subdirs)} dataset subdirectories.")
    tasks = []

    for subdir in dataset_subdirs:
        domain_problem_pairs = get_all_domain_problem_pairs(subdir)
        for domain_file, problem_file in domain_problem_pairs:
            domain_name = subdir.name
            task_number = problem_file.stem.split("task")[1]
            if not validate_task_inputs(domain_file, problem_file, logger):
                continue
            for search_algorithm in ['astar', 'gbf']:
                for heuristic in ['lmcut', 'hff']:
                    output_path = root_dir / "csv_output" / f"{search_algorithm}_{heuristic}_{domain_name}_{task_number}.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    tasks.append((domain_file, problem_file, search_algorithm, heuristic, True, output_path))
    return tasks

def main():
    logger = setup_logger()
    logger.info("Starting the planning process.")

    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "dataset"

    tasks = prepare_tasks(dataset_dir, root_dir, logger)
    logger.info(f"Prepared {len(tasks)} tasks for execution.")

    max_workers = os.cpu_count() or 4
    logger.info(f"Using {max_workers} workers.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_search_task_with_timeout, task, logger): task for task in tasks}
        try:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Task succeeded: {task}")
                    else:
                        logger.info(f"Task did not produce a result: {task}")
                except Exception as e:
                    logger.error(f"Unexpected error in task {task}: {e}", exc_info=True)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected. Attempting to shut down remaining tasks.")
            executor.shutdown(wait=False)
        finally:
            logger.info("All tasks processed.")

if __name__ == "__main__":
    main()
