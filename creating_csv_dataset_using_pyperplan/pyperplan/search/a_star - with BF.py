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

"""
Implements the A* (a-star) and weighted A* search algorithm.
"""
import csv
import heapq
import logging

from . import searchspace
import sys


def ordered_node_astar(node, h, node_tiebreaker):
    """
    Creates an ordered search node (basically, a tuple containing the node
    itself and an ordering) for A* search.

    @param node The node itself.
    @param heuristic A heuristic function to be applied.
    @param node_tiebreaker An increasing value to prefer the value first
                           inserted if the ordering is the same.
    @returns A tuple to be inserted into priority queues.
    """
    f = node.g + h
    return (f, h, node_tiebreaker, node)


def ordered_node_weighted_astar(weight):
    """
    Creates an ordered search node (basically, a tuple containing the node
    itself and an ordering) for weighted A* search (order: g+weight*h).

    @param weight The weight to be used for h
    @param node The node itself
    @param h The heuristic value
    @param node_tiebreaker An increasing value to prefer the value first
                           inserted if the ordering is the same
    @returns A tuple to be inserted into priority queues
    """
    """
    Calling ordered_node_weighted_astar(42) actually returns a function (a
    lambda expression) which is the *actual* generator for ordered nodes.
    Thus, a call like
        ordered_node_weighted_astar(42)(node, heuristic, tiebreaker)
    creates an ordered node with weighted A* ordering and a weight of 42.
    """
    return lambda node, h, node_tiebreaker: (
        node.g + weight * h,
        h,
        node_tiebreaker,
        node,
    )


def ordered_node_greedy_best_first(node, h, node_tiebreaker):
    """
    Creates an ordered search node (basically, a tuple containing the node
    itself and an ordering) for greedy best first search (the value with lowest
    heuristic value is used).

    @param node The node itself.
    @param h The heuristic value.
    @param node_tiebreaker An increasing value to prefer the value first
                           inserted if the ordering is the same.
    @returns A tuple to be inserted into priority queues.
    """
    f = h
    return (f, h, node_tiebreaker, node)


def greedy_best_first_search(task, heuristic, use_relaxed_plan=False):
    """
    Searches for a plan in the given task using greedy best first search.

    @param task The task to be solved.
    @param heuristic A heuristic callable which computes the estimated steps
                     from a search node to reach the goal.
    """
    return astar_search(
        task, heuristic, ordered_node_greedy_best_first, use_relaxed_plan
    )


def weighted_astar_search(task, heuristic, weight=5, use_relaxed_plan=False):
    """
    Searches for a plan in the given task using A* search.

    @param task The task to be solved.
    @param heuristic  A heuristic callable which computes the estimated steps.
                      from a search node to reach the goal.
    @param weight A weight to be applied to the heuristics value for each node.
    """
    return astar_search(
        task, heuristic, ordered_node_weighted_astar(weight), use_relaxed_plan
    )


def astar_search(
    task, heuristic, make_open_entry=ordered_node_astar, use_relaxed_plan=False
):
    """
    Searches for a plan in the given task using A* search.

    @param task The task to be solved
    @param heuristic  A heuristic callable which computes the estimated steps
                      from a search node to reach the goal.
    @param make_open_entry An optional parameter to change the bahavior of the
                           astar search. The callable should return a search
                           node, possible values are ordered_node_astar,
                           ordered_node_weighted_astar and
                           ordered_node_greedy_best_first with obvious
                           meanings.
    """
    open_list = []
    state_cost = {task.initial_state: 0}
    node_tiebreaker = 0


    root = searchspace.make_root_node(task.initial_state)
    init_h = heuristic(root)
    heapq.heappush(open_list, make_open_entry(root, init_h, node_tiebreaker))
    logging.info("Initial h value: %f" % init_h)

    besth = float("inf")
    counter = 0
    expansions = 0
    resutls = []
    cnt = 0
    node_dict = {}
    node_values = {}
    output_name = sys.argv[-1]
    while open_list:
        (f, h, _tie, pop_node) = heapq.heappop(open_list)

        if h < besth:
            besth = h
            logging.debug("Found new best h: %d after %d expansions" % (besth, counter))

        pop_state = pop_node.state
        #if node_dict.get(pop_node.state) == None: 
        node_dict[pop_node.state] = cnt
        if cnt == 0: # if i am root
            father_n = None
        else: # if i am not root
            father_n = node_dict[pop_node.parent.state]
        node_values[cnt] = ((father_n, f, h, f-h)) # (parent#, f, h, g)

        if (pop_node.parent == None): # i dont have father and i am root
            father_n = 0
            father_f = 0
            father_h = 0
            father_g = 0
            grandfather_n = 0
            grandfather_f = 0
            grandfather_h = 0
            grandfather_g = 0
        else:
            father_n = node_values[cnt][0]
            father_f = node_values[cnt][1]
            father_h = node_values[cnt][2]
            father_g = node_values[cnt][3]

            if node_values[cnt][0] == 0: # my father is rrot
                grandfather_n = 0
                grandfather_f = 0
                grandfather_h = 0
                grandfather_g = 0
            else: # i have grandfather
                grandfather_n = node_values[father_n][0]
                grandfather_f = node_values[father_n][1]
                grandfather_h = node_values[father_n][2]
                grandfather_g = node_values[father_n][3]                
        resutls.append((cnt, f, h, f-h, father_n, father_f, father_h, father_g, grandfather_n, grandfather_f, grandfather_h, grandfather_g, len(task.get_successor_states(pop_state))))

        # Only expand the node if its associated cost (g value) is the lowest
        # cost known for this state. Otherwise we already found a cheaper
        # path after creating this node and hence can disregard it.
        if state_cost[pop_state] == pop_node.g:
            expansions += 1

            if task.goal_reached(pop_state):
                with open(output_name+'.csv', 'w', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    temp = ["node#", "node_f", "node_h", "node_g", "father_node#", "father_node_f", "father_node_h", "father_node_g",
                            "grandfather_node#", "grandfather_node_f", "grandfather_node_h", "grandfather_node_g",  "node_state"]
                    wr.writerow(temp)
                    for state in resutls:
                        wr.writerow(state)
                logging.info("Goal reached. Start extraction of solution.")
                logging.info("%d Nodes expanded" % expansions)
                return pop_node.extract_solution()
            rplan = None
            if use_relaxed_plan:
                (rh, rplan) = heuristic.calc_h_with_plan(
                    searchspace.make_root_node(pop_state)
                )
                logging.debug("relaxed plan %s " % rplan)
            #print("task.get_successor_states(pop_state) = ", len(task.get_successor_states(pop_state)))
            for op, succ_state in task.get_successor_states(pop_state):
                #print("op = ", op)
                if use_relaxed_plan:
                    if rplan and not op.name in rplan:
                        # ignore this operator if we use the relaxed plan
                        # criterion
                        logging.debug(
                            "removing operator %s << not a "
                            "preferred operator" % op.name
                        )
                        continue
                    else:
                        logging.debug("keeping operator %s" % op.name)

                succ_node = searchspace.make_child_node(pop_node, op, succ_state)

                h = heuristic(succ_node)
                if h == float("inf"):
                    # don't bother with states that can't reach the goal anyway
                    continue
                old_succ_g = state_cost.get(succ_state, float("inf"))
                if succ_node.g < old_succ_g:
                    # We either never saw succ_state before, or we found a
                    # cheaper path to succ_state than previously.
                    node_tiebreaker += 1
                    heapq.heappush(open_list, make_open_entry(succ_node, h, node_tiebreaker))
                    state_cost[succ_state] = succ_node.g

        counter += 1
        cnt += 1
    logging.info("No operators left. Task unsolvable.")
    logging.info("%d Nodes expanded" % expansions)
    return None
