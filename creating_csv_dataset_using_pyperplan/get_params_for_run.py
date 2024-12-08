from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def get_all_dirs_in_dir(dir_path):
    """
    Retrieves all directories in a given directory.
    Args:
        dir_path (Path): The directory path where the directories are located.
    Returns:
        list of Path: A list of paths to the directories in the specified directory.
    """
    return [path for path in dir_path.iterdir() if path.is_dir()]

def get_all_domain_problem_pairs(dir_path):
    """
    Retrieves all pairs of domain and problem files from a given directory.
    This function searches for files in the specified directory that match the
    patterns "domain*.pddl" and "task*.pddl". If exactly one domain file is found
    and one or more problem files are found, it pairs the single domain file with
    each problem file. Otherwise, it pairs domain files and problem files in the
    order they are found.
    Args:
        dir_path (Path): The directory path where the domain and problem files are located.
    Returns:
        list of tuples: A list of tuples where each tuple contains a domain file and a problem file.
    """

    domain_files = list(dir_path.glob("domain*.pddl"))
    problem_files = list(dir_path.glob("task*.pddl"))
    if len(domain_files) == 1 and len(problem_files) > 0:
        domain_file = domain_files[0]
        return [(domain_file, problem_file) for problem_file in problem_files]
    else:
        return [(domain_file, problem_file) for domain_file, problem_file in zip(domain_files, problem_files)]

if __name__ == "__main__":
    domain_problem_pairs = get_all_domain_problem_pairs(ROOT_DIR / "dataset" / "blocks")
    for domain_file, problem_file in domain_problem_pairs:
        print(f"Domain file: {domain_file}, Problem file: {problem_file}")