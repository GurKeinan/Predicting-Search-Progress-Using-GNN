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


def greedy_best_first_search(task, heuristic, output_path, use_relaxed_plan=False):
    """
    Searches for a plan in the given task using greedy best first search.

    @param task The task to be solved.
    @param heuristic A heuristic callable which computes the estimated steps
                     from a search node to reach the goal.
    """
    return astar_search(
        task, heuristic, output_path, ordered_node_greedy_best_first, use_relaxed_plan
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
    task, heuristic, output_path, make_open_entry=ordered_node_astar, use_relaxed_plan=False
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
    cnt = 1
    node_dict = {}
    node_values = {}
    output_name = output_path
    h0 = 0
    H_min = None
    last_H_min_update= 0
    f_max = None
    while open_list:
        (f, h, _tie, pop_node) = heapq.heappop(open_list)
        g = pop_node.g
        if h < besth:
            besth = h
            logging.debug("Found new best h: %d after %d expansions" % (besth, counter))

        pop_state = pop_node.state
        #if node_dict.get(pop_node.state) == None:
        node_dict[pop_node.state] = ((cnt,len(task.get_successor_states(pop_state)))) # dict from state -> (node#, BF)
        if cnt == 1: # if i am root
            father_n = None
        else: # if i am not root
            father_n = node_dict[pop_node.parent.state][0] # father node#
        if (cnt > 1000000):
            print("more then 1m nodes")
            return None
        BF = len(task.get_successor_states(pop_state))
        node_values[cnt] = ((father_n, f, h, g, BF)) # (parent#, f, h, g, BF)

        temp_cnt = cnt
        number_of_nodes = 10 # number of nodes
        number_of_features = 5 # number of features per node
        matrix = [[0 for x in range(number_of_features)] for y in range(number_of_nodes)]
        for i in range(number_of_nodes):
            if temp_cnt == None:
                for j in range(number_of_features):
                    matrix[i][j] = 0
            else:
                matrix[i][0] = temp_cnt
                matrix[i][1] = node_values[temp_cnt][1]
                matrix[i][2] = node_values[temp_cnt][2]
                matrix[i][3] = node_values[temp_cnt][3]
                matrix[i][4] = node_values[temp_cnt][4]
                if node_values[temp_cnt][0] == None:
                    temp_cnt = None
                else:
                    temp_cnt = node_values[temp_cnt][0]

        if (cnt == 1):
            h0 = h
        if (H_min == None or H_min>h):
            H_min = h
            last_H_min_update = cnt
        if (f_max == None or f_max < f):
            f_max = f
        #resutls.append((cnt, BF, f, h, f-h, father_n, father_BF, father_f, father_h, father_g, grandfather_n, grandfather_BF,grandfather_f, grandfather_h, grandfather_g, h0, H_min, cnt - last_H_min_update, f_max))
        # resutls.append((
        #     matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3], matrix[0][4],
        #     matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3], matrix[1][4],
        #     matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3], matrix[2][4],
        #     matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3], matrix[3][4],
        #     matrix[4][0], matrix[4][1], matrix[4][2], matrix[4][3], matrix[4][4],
        #     matrix[5][0], matrix[5][1], matrix[5][2], matrix[5][3], matrix[5][4],
        #     matrix[6][0], matrix[6][1], matrix[6][2], matrix[6][3], matrix[6][4],
        #     matrix[7][0], matrix[7][1], matrix[7][2], matrix[7][3], matrix[7][4],
        #     matrix[8][0], matrix[8][1], matrix[8][2], matrix[8][3], matrix[8][4],
        #     matrix[9][0], matrix[9][1], matrix[9][2], matrix[9][3], matrix[9][4],
        #     h0, H_min, cnt - last_H_min_update, f_max
        # ))
        resutls.append((
            matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3], matrix[0][4],
            matrix[1][0], h0, H_min, cnt - last_H_min_update, f_max
        ))

        # Only expand the node if its associated cost (g value) is the lowest
        # cost known for this state. Otherwise we already found a cheaper
        # path after creating this node and hence can disregard it.
        if state_cost[pop_state] == pop_node.g:
            expansions += 1

            if task.goal_reached(pop_state):
                with open(output_name, 'w', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # temp = [
                    #     "level_1#", "level_1_F", "level_1_H", "level_1_G", "level_1_BF",
                    #     "level_2#", "level_2_F", "level_2_H", "level_2_G", "level_2_BF",
                    #     "level_3#", "level_3_F", "level_3_H", "level_3_G", "level_3_BF",
                    #     "level_4#", "level_4_F", "level_4_H", "level_4_G", "level_4_BF",
                    #     "level_5#", "level_5_F", "level_5_H", "level_5_G", "level_5_BF",
                    #     "level_6#", "level_6_F", "level_6_H", "level_6_G", "level_6_BF",
                    #     "level_7#", "level_7_F", "level_7_H", "level_7_G", "level_7_BF",
                    #     "level_8#", "level_8_F", "level_8_H", "level_8_G", "level_8_BF",
                    #     "level_9#", "level_9_F", "level_9_H", "level_9_G", "level_9_BF",
                    #     "level_10#", "level_10_F", "level_10_H", "level_10_G", "level_10_BF",
                    #     "H0", "H_min", "last_H_min_update", "f_max","node_max"
                    #         ]
                    temp = ["serial", "f", "h", "g", "BF", "father_serial", "h0", "H_min", "last_H_min_update", "f_max", "node_max"]
                    wr.writerow(temp)
                    for state in resutls:
                        new_state = state + (cnt,)
                        wr.writerow(new_state)
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
