#! /usr/bin/env python3
#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

# TODO: Give searches and heuristics commandline options and reenable preferred
# operators.

import argparse
import logging
import os
import sys

from pyperplan import tools
from pathlib import Path
from pyperplan.planner import (
    find_domain,
    HEURISTICS,
    search_plan,
    SEARCHES,
    validate_solution,
    write_solution,
)

# possible search algorithms: astar, wastar, gbf, bfs, ehs, ids, sat
# possible heuristics: landmark, lmcut, hadd, hff, hmax, hsa, blind


def main():
    # Hardcoded parameters
    root_dir = Path(__file__).resolve().parent.parent
    domain_file = root_dir / "dataset" / "blocks" / "domain.pddl"
    problem_file = root_dir / "dataset" / "blocks" / "task18.pddl"
    search_algorithm = "wastar"
    heuristic = "lmcut"
    output_file = "output_plan.soln"
    validate = True

    # Set log level
    # tools.setup_logging(loglevel)

    # Find domain and problem files
    domain_file = os.path.abspath(domain_file)
    problem_file = os.path.abspath(problem_file)

    # Run the search algorithm
    heuristic_fn = HEURISTICS[heuristic]
    search_fn = SEARCHES[search_algorithm]
    plan = search_plan(domain_file, problem_file, search_fn, heuristic_fn)

    # Validate the solution if requested
    if validate:
        validate_solution(domain_file, problem_file, plan)

    # Write the solution to the output file if specified
    if output_file:
        write_solution(plan, output_file)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
