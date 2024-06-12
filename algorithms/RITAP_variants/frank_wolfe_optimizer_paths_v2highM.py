import numpy as np
import argparse
import networkit as nk
import numpy as np
import networkx as nx
from scipy import stats
import copy
import re
import time
import os
import json


# this program inplements the Frank-Wolfe method to solve the

FLOW_EPS = 1e-8  # flows smaller than this value will be set to zero
DERIV_EPS = 1e-10  # small number used to compute the derivative of the activation function near zero, whener the derivative is infinite at zero.


def phi(x, gamma):
    
    alpha=0.15
    cap=20
    #return x+alpha*(x**(gamma+1))/((gamma+1)*(cap**gamma))
    return x*(1+alpha/(gamma+1)*(x/cap)**gamma)
    # return x**gamma


def deriv_phi(x, gamma):
    alpha=0.15
    cap=20
    return 1+alpha*(x/cap)**gamma
    #return gamma * (x + DERIV_EPS) ** (gamma - 1)

def copy_graph(G, reset_weights=False, weight_value=0):
    """
    returns a copy of a networkit graph G
    if reset_weights==True, and the graph is weighted, then all the weights in the graph are set to 'weight_value'.
    if reset_weight==False, and the graph is weighted, then the weights of G are copied
    """
    N = G.numberOfNodes()
    G_copy = nk.graph.Graph(
        n=G.numberOfNodes(),
        weighted=G.isWeighted(),
        directed=G.isDirected(),
        edgesIndexed=False,
    )
    for u, v in G.iterEdges():
        G_copy.addEdge(u, v)
        if G.isWeighted():
            if reset_weights:
                G_copy.setWeight(u, v, weight_value)
            else:
                G_copy.setWeight(u, v, G.weight(u, v))
    return G_copy

def RGAP(G,G_traf,bushes):
    """implements the RGAP function as defined in 'O. Perederieieva et al, A framework for and empirical study of algorithms for traffic assignment.'.
    
    the equivalent of c_a is the weight of edge a in graph G.
    G_traf: a networkit weighted graph. the weight on each edge e is equal to the total traffic I_e on that edge
    G: a networkit weighted graph. the weight on each edge e is equal to phi'(I_e), where I_e is the total traffic on edge e and phi' is the derivative of phi.    
    start_nodes: list containing the starting nodes of the paths
    end_nodes: list containing the end nodes of the paths. Is expected to ob of the same length as start_nodes
    """
    FW_deriv_cost=sum([G_traf.weight(u,v)*w for u, v, w in G.iterEdgesWeights()])
    SP_G_deriv_cost=0
    
    for start in bushes:
        dijkstra = nk.distance.Dijkstra(G, source=start)
        dijkstra.run()
        for end in bushes[start]:
            SP_G_deriv_cost+=dijkstra.distance(end)*bushes[start][end]

        
    return 1- SP_G_deriv_cost/FW_deriv_cost


def is_file_empty(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        return os.path.getsize(file_path) == 0
    else:
        # File doesn't exist, treat it as empty
        return True


def starts_ends_to_origin_bush(start_nodes, end_nodes):
    """
    This function converts the two lists start_nodes and end_nodes into a different data structure that collects all the paths with a commom start node.


    A bush is a set of start-end pairs with a common start node. The data structure to represent many bushes is the following

    {s1 : {e1_s1 : w_s1e1, e2_s1 : w_s1e2,...}, s2 : {e1_s2 : w_s2e1, e2_s2 : w_s2e2,...}, ...}
    s are the start nodes
    e_s are the various end nodes corresponding to a start node s
    w_se is the number of paths going from s to e.
    The advantage of this data structure is the following: dijkstra computes the shortest paths from one start node to all possible end nodes. Therefore grouping the paths by start node we reduce the number fo calls to dijkstra.
    The ordering of the paths is lost in this new representation.
    """
    bushes = {}

    for start, end in zip(start_nodes, end_nodes):
        if start in bushes:
            if end in bushes[start]:
                bushes[start][end] = bushes[start][end] + 1
            else:
                bushes[start][end] = 1
        else:
            bushes[start] = {end: 1}

    return bushes


def FW_traffic_assignment_single_commodity(bushes, G, lr, tmax, phi, deriv_phi, rtol=1e-7, rel_opt_gap_thr=0):
    """
    Uses the Frank-Wolfe algorithm to compute the continuous traffic equilibrium that minimizes H(I)=sum_e phi(I_e), where I_e is the total traffic on edge e.

    Arguments:

    start_nodes: list containing the starting nodes of the paths
    end_nodes: list containing the end nodes of the paths. Is expected to ob of the same length as start_nodes
    G: weighted networkit graph. G's weights will be modified by the algorithm
    lr (list): learning rate of the Frank-Wolfe algorithm. To have convergence, lr[t] shoudl go to zero when t becomes large. It is expcted that len(lr)=tmax
    phi (function): activation function to define the energy H
    phi_deriv (function): derivative of phi
    tmax (int): maximum number of Frank-Wolfe iterations
    rtol (float): relative tolerance before declaring convergence. if |H(t)-H(t-1)|/H(t-1)<rtol]lr[t], then the algorithm halts. H(t) here is the value of the energy after t iterations.
    rel_opt_gap_thr: alternative halting condition based on the otimality gap. When |optimality_gaps[t]|/H(t)<rel_opt_gap_thr, the algorithm halts. When phi is not convex, always set rel_opt_gap_thr=0.


    WARNING: This algorithm will not work for directed graphs.
    Returns:
    G_traf: a networkit weighted graph. the weight on each edge e is equal to the total traffic I_e on that edge
    G: a networkit weighted graph. the weight on each edge e is equal to phi'(I_e), where I_e is the total traffic on edge e and phi' is the derivative of phi.
    dict_paths: this dictionary contains each commodity 's flow on the graph.
    energies (list): the list of values of H encounteres at every time step of the algorithm.
    optimality_gaps (list): if phi is convex, then energies[t]-optimality_gaps[t] is a lower bound to the energy of the optimal solution. THIS IS EXCLUSIVELY VALID WHEN PHI IS CONVEX, otherwise this quantity is meaningless.
    conv_flag (bool): if True the algorithm has halted because it satisfied one of the convergence conditions, if False it halted when it finished tmax iterations
    final_t (integer): the iteration number at which the algorithm converged
    """

    # convert start_nodes, end_nodes to bush data structure
    #N = G.numberOfNodes()
    #M = len(start_nodes)
    energies = []
    optimality_gaps = []

    # reset weights in G
    for u, v in G.iterEdges():
        G.setWeight(u, v, 1)

    dict_paths = {} # this nested dictionary contains the flows for every start-end pair. If there are multiple paths with the same start-end pair, their combined flow is stored in the same entry of the dictionary.
    # keys of this dictionaries are tuples with this structure: (start,end), (edge[0],edge[1]). The first pair of keys indexes the top level dictonary, while the second pair of keys indexes the second level dictionary.
    # start is the starting node , end is the final node, edge[0], edge[1] are the two nodes joined by the edge.
    # the convention is that edge[0]<edge[1]. If one wishes to use directed edges, the use of this data structure should be modified.
    # dict_paths[start,end][edge[0],edge[1]] stores the traffic of the flow going from start to end that traverses the edge (edge[0],edge[1]).
    G_traf = copy_graph(G, reset_weights=True, weight_value=0)
    for start in bushes:
        dijkstra = nk.distance.Dijkstra(G, source=start)
        dijkstra.run()
        for end in bushes[start]:
            dict_paths[start, end] = {}
            path = dijkstra.getPath(end)
            for k in range(len(path) - 1):
                dict_paths[start, end][min(path[k], path[k + 1]), max(path[k], path[k + 1])] = bushes[start][end]
                G_traf.setWeight(path[k],path[k + 1],G_traf.weight(path[k], path[k + 1]) + bushes[start][end])

    # auxiliary graph where the all-or-nothing (aon) flows will be stored. These are called all-or-nothing because all the flow between a start and an end node is routed through a single path.
    G_traf_aon = copy_graph(G, reset_weights=True, weight_value=0)
    conv_flag = False
    for t in range(tmax):
        # measure_energy
        total_energy = sum([phi(w) for _, _, w in G_traf.iterEdgesWeights()])
        energies.append(total_energy)
        # convergence condition
        if t > 1 and (abs(energies[t] - energies[t - 1]) / energies[t - 1] < rtol * lr[t] or abs(opt_gap) / energies[t - 1] < rel_opt_gap_thr):
            conv_flag = True
            optimality_gaps.append(opt_gap)
            break

        # compute the gradient
        for u, v in G.iterEdges():
            G.setWeight(u, v, deriv_phi(G_traf.weight(u, v)))
            G_traf_aon.setWeight(u, v, 0)

        for key in dict_paths:
            for key2 in dict_paths[key]:
                dict_paths[key][key2] = dict_paths[key][key2] * (1 - lr[t])

        # compute all-or-nothing solution
        # use dijkstra to minimize
        for start in bushes:
            dijkstra = nk.distance.Dijkstra(G, source=start)
            dijkstra.run()
            for end in bushes[start]:
                path = dijkstra.getPath(end)
                for k in range(len(path) - 1):
                    G_traf_aon.setWeight(
                        path[k],
                        path[k + 1],
                        G_traf_aon.weight(path[k], path[k + 1]) + bushes[start][end],
                    )
                    if (min(path[k], path[k + 1]),max(path[k], path[k + 1])) in dict_paths[start, end]:
                        dict_paths[start, end][min(path[k], path[k + 1]), max(path[k], path[k + 1])] = (dict_paths[start, end][min(path[k], path[k + 1]), max(path[k], path[k + 1])]+ lr[t] * bushes[start][end]
                        )
                    else:
                        dict_paths[start, end][min(path[k], path[k + 1]), max(path[k], path[k + 1])] = (lr[t] * bushes[start][end])

        # making a step towards G_traf_aon
        opt_gap = 0
        for u, v in G_traf.iterEdges():
            opt_gap -= G.weight(u, v) * (G_traf_aon.weight(u, v) - G_traf.weight(u, v))
            G_traf.setWeight(u,v,(1 - lr[t]) * G_traf.weight(u, v) + lr[t] * G_traf_aon.weight(u, v))

        optimality_gaps.append(opt_gap)

    return G_traf, G, dict_paths, energies, optimality_gaps, conv_flag, t


def networkit_to_networkx_graph(G):
    """
    Converts networkit graph to networkx graph. Also copies the weights in case of weighted graph.
    """
    if (G.isDirected()):
        G_nx = nx.DiGraph()
    else:
        G_nx = nx.Graph()
      
    
    for i in range(G.numberOfNodes()):
        G_nx.add_node(i)
    if(G.isWeighted()):
        for u,v in G.iterEdges():
            G_nx.add_edge(u,v,weight=(G.weight(u,v)))
    else:
        for u,v in G.iterEdges():
            G_nx.add_edge(u,v)
            
    return G_nx


def support_paths(G,bushes,atol=0, rtol=1e-4):
    """
    Computes the paths on which positive traffic can be allocated.
    
    Arguments:
    G: networkit graph. It is assumed that G is weighted with phi'(I_e), where I_e is the flow on edge e and phi' is the derivative of the activation function phi. {I_e}_{e \in E} is the flow that minimizes the energy H(I)=sum_e phi(I_e). 
    bushes: the demand matrix in dictionary form

    The support paths are the paths on which positive traffic can be allocated. The name 'support_paths' comes from the fact that the traffic on these paths supports the optimal flow (and from the analogy with support vector machines).
      The theory of the traffic assignment problem (see for example Transportation Network Analysis Volume I: Static and Dynamic Traffic Assignment. Boyles, Lownes, Unnikrishnan) guarantees that the optimal flow is routed exclusively on the shortest paths with respect to the graph G. 
    However, since the optimizer does not exactly reach this flow, this relation is not guaranteed to hold. Therefore we check it with a tolerance.
    
    For a given start-end pair, the function uses Yen's k-shortest path algorithm to enumerate the paths between start end end in order of increasing cost (as measured with respect to the weights in graph G).
    Calling path_cost the cost of the path being considered, and sp_cost the cost of the shortest path between start and end, the algorithm adds path to the list of support paths if  |path_cost-sp_cost|<=atol+rtol*sp_cost.
    Otherwise the algorithm stops enumerating paths for that start-end pair.

    Returns:
    support_paths_dict: a dictionary containing the support paths for each start-end pair. support_paths_dict[start,end] is the list of support paths starting at 'start' and terminating in 'end'. Each path is a list of nodes.
    
    """
    G_nx=networkit_to_networkx_graph(G) #converting G to a networkx graph, since the library to computethe k shortest paths works only for networkx graphs.
    support_paths_dict={}
    for start in bushes:
        for end in bushes[start]:
            supp_paths_se=[]
            shortest_paths_se_gen=nx.shortest_simple_paths(G_nx, start, end, weight='weight')
            flag_sp=True
            for path in shortest_paths_se_gen:
                path_cost=sum([G.weight(path[k],path[k+1]) for k in range(len(path)-1)])
                if flag_sp:
                    sp_cost=path_cost
                    flag_sp=False
                
                elif(path_cost-sp_cost>atol+rtol*sp_cost): 
                    break   
                supp_paths_se.append(path.copy())
        
            support_paths_dict[start,end]=copy.deepcopy(supp_paths_se)
        
    return support_paths_dict


def single_flow_to_paths(G_traf, start, end, flow_eps=0, maxiter=None):
    """
    Given a single commodity flow, between a source node 'start' and an end node 'end', this algorithm finds a path decomposition of this flow.

    G_traf: weighted networkit graph containing the flow on each edge of the graph
    start: start node
    end: end node
    flow_eps: (float) flows smaller than flow_eps will be set to zero. It is advisable to set it to something like 1e-10.
    maxiter: (int) maximum number of iterations default is None, corresponding to infinite limit.

    Returns
    path_list: a list in which every element is a path. A path is a list containing the sequence of nodes traversed by the path, starting from the start node and finishing with the end node.
    path_traf_list: for every path this list specifies the traffic carried by that path. path_list[i] brings traffic path_traf_list[i].
    """

    BIGNUM = 1e100
    incr = 1
    if maxiter is None:
        maxiter = 1
        incr = 0

    a = 0.001  # the smaller a, the more likely it is that one finds paths carrying more traffic (as opposed to shorter paths) first (when runninng the loop)
    G_aux = copy_graph(
        G_traf, reset_weights=True, weight_value=BIGNUM
    )  # edges with a lot of flow in G_traf will have a small cost in G_aux. We use dijkstra on G_aux to select the paths. This way we will first select the paths carrying the most flow.
    for u, v, w in G_traf.iterEdgesWeights():
        if w > flow_eps:
            G_aux.setWeight(u, v, 1 / (w + a))
    if start == end:
        print("ERROR start node should be different from end node")
        return 0

    path_list = []
    path_traf_list = []
    iter_count = 0
    while iter_count < maxiter:
        dijkstra = nk.distance.Dijkstra(G_aux, source=start)
        dijkstra.run()
        path = dijkstra.getPath(end)
        path_traf = min(
            [G_traf.weight(path[k], path[k + 1]) for k in range(len(path) - 1)]
        )
        if dijkstra.distance(end) > 0.5 * BIGNUM or path_traf < flow_eps:
            break

        for k in range(len(path) - 1):
            G_traf.setWeight(path[k], path[k + 1], G_traf.weight(path[k], path[k + 1]) - path_traf)  # subtracting the contribution of the path from the total flow
            if G_traf.weight(path[k], path[k + 1]) <= flow_eps:
                G_aux.setWeight(path[k], path[k + 1], BIGNUM)
                G_traf.setWeight(path[k], path[k + 1], 0)

            else:
                G_aux.setWeight(path[k], path[k + 1], 1 / (a + G_traf.weight(path[k], path[k + 1])))

        path_list.append(path.copy())
        path_traf_list.append(path_traf)
        iter_count += incr

    path_traf_list, path_list = (list(t) for t in zip(*sorted(zip(path_traf_list, path_list), reverse=True)))
    return path_list, path_traf_list


def flows_to_paths(G_traf, dict_paths, bushes, flow_eps=1e-12):
    """
    Converts the flow into (noninteger) paths.
    G_traf: networkit graph whose edges are weighted with I_e, the flow through the edge.
    dict_paths: dictionary containing the traffic on each edge for every commodity. dict_paths[start,end]={e1: f1,e2:f2,...} where start,end are the starting and ending ponit of the path. e1=(e10,e11) is an edge is the graph, f1,f2,... are the flows on the respective edges.
    Notice that if there are npaths with the same start and end point the combined flow is stored in a single enrty of the dicrionary.
    start_nodes: list containing the start nodes for each path
    end_nodes: list containing the end nodes for each path
    flow_eps: flows smaller that flow_eps are set to zero


    Returns
    paths_dict. paths_dict[start,end]=[[p1,p2,...],[t1,t2,..]], where paths p1,p2,... are the paths between start and end that carry the traffic. t1,t2,... are the traffic on the respective paths. each of p1,p2,... is a list contai ing the nodes traversed by the path.
    """
    paths_dict = {}
    for start in bushes:
        for end in bushes[start]:
            G_traf_se = copy_graph(G_traf, reset_weights=True, weight_value=0)  # graph storing the flow for all the paths with common start point and end point
            for u, v in dict_paths[start, end]:
                if (dict_paths[start, end][u, v] > flow_eps):  # setting the flow to zero if it's too small
                    G_traf_se.setWeight(u, v, dict_paths[start, end][u, v])
            paths_se, traffic_paths_se = single_flow_to_paths(G_traf_se, start, end, flow_eps=flow_eps)
            paths_dict[start, end] = [copy.deepcopy(paths_se), traffic_paths_se.copy()]

    return paths_dict


def continuous_to_max_traf_integer_paths(paths_dict, bushes):
    """
    Computes the integer paths starting from the paths carrying a noninteger traffic. When the traffic is split across many paths, the algorithm chooses the path with the largest traffic.

    For each start-end pair, the algorithm puts all the traffic on the path that is carrying the most traffic. For example if a traffic of 1 is split among three paths in the following proportions :0.45,0.35,0.2, the algorithm will choose the first path (carrying 0.45) as integer path.
    The algorithm is conceived to handle correctly the case when there are multiple paths with the same start and end nodes.
    Arguments:
    paths_dict[start,end]=[[p1,p2,...],[t1,t2,..]], where paths p1,p2,... are the paths between start and end that carry the traffic. t1,t2,... are the traffic on the respective paths. each of p1,p2,... is a list contai ing the nodes traversed by the path.
    bushes: bushes representation of the demand matrix

    Returns:
    a bush (nested dictionary whose keys are the start and end node) integer_paths_bush, such that integer_paths_bush[start][end]=[[pi_1,...,pi_R],[tINT_1,...,tINT_R], [t_1,...,t_R]] with pi_i paths from start to end carrying an integer amount of flow tINT_i and originally carrying an amount of flow t_i in the continuous solution
    """
    integer_paths_dict ={} #bush (nested dictionary) containing the paths and flow for each start-end pair
    for start,end in paths_dict:
        if integer_paths_dict.get(start) is None:
            integer_paths_dict[start]={}
        path_list=copy.deepcopy(paths_dict[start, end][0])
        path_traf_list=copy.copy(paths_dict[start,end][1])
        integer_paths_dict[start][end]=[[],[],[]]
        allocated_flow=0
        for i,traf in enumerate(path_traf_list):
            if(traf>=1): #adding the paths with more than one unit of flow
                integer_paths_dict[start][end][0].append(path_list[i])
                integer_paths_dict[start][end][1].append(int(traf))
                integer_paths_dict[start][end][2].append(int(traf))
                allocated_flow+=int(traf)
                path_traf_list[i]=path_traf_list[i]%1 #keeping the fractional part

        if(allocated_flow<bushes[start][end]): #now adding the fractional parts if necessary
            path_traf_list, path_list = (list(t) for t in zip(*sorted(zip(path_traf_list, path_list), reverse=True))) # sorting paths in order of decreasing fractional traffic
            for k in range(bushes[start][end]-allocated_flow):
                if(path_list[k] in integer_paths_dict[start][end][0]):#if the path is already present in the list we increment its 
                    idx=integer_paths_dict[start][end][0].index(path_list[k])
                    integer_paths_dict[start][end][1][idx]+=1
                    integer_paths_dict[start][end][2][idx]+=path_traf_list[k]

                else:# if the path is not present we append it to the list of paths
                    integer_paths_dict[start][end][0].append(path_list[k])
                    integer_paths_dict[start][end][1].append(1)
                    integer_paths_dict[start][end][2].append(path_traf_list[k])

    return integer_paths_dict

def misallocated_traffic_optimum(G, paths_dict, rtol=5e-2, atol=0, quiet=False):
    """
    Computes the total traffic that is allocated on paths that, up to some tolerance, are incompatible with the fact that at the optimal solution the flow is routed exclusively over the shortest paths with respect to graph G.
    I theory this relation holds at the optimum flow. However, since the optimizer does not exactly reach this flow, this relation is not guaranteed to hold. Therefore we check it with a tolerance.
    For theory reference see Transportation Network analysis, Boyles et al., The traffic Assignment Problem models and methods, Patriksson.
    G is a graph with weights equal to phi'(I_e), where I_e is the flow on edge e and phi' is the derivative of the activation function phi.
    paths_dict is a dictionary containing the paths over which the flow is allocated. paths_dict[start,end]=[[p1,p2,...],[t1,t2,..]], where paths p1,p2,... are the paths between start and end that carry the traffic. t1,t2,... are the traffic on the respective paths. each of p1,p2,... is a list contai ing the nodes traversed by the path.
    Returns the total traffic that is allocated on paths that do not satisfy the following condition: cost(path)-min_cost(start,end)<=rtol*min_cost(start,end)+atol. start,end are the endpoints of path.
    Also prints a warning if the condition is not satisfied.
    """
    tot_misallocated_traffic = 0
    for start, end in paths_dict:
        paths_se = paths_dict[start, end][0]
        traffics = paths_dict[start, end][1]
        dijkstra = nk.distance.Dijkstra(G, source=start)
        dijkstra.run()
        path_min_cost_G = dijkstra.distance(end)
        for i in range(len(traffics)):
            path = paths_se[i]
            path_cost_G = sum([G.weight(path[k], path[k + 1]) for k in range(len(path) - 1)])
            discrep = path_cost_G - path_min_cost_G
            if discrep > atol + path_min_cost_G * rtol:
                tot_misallocated_traffic += traffics[i]
                if not quiet:
                    print(f"WARNING: traffic of {traffics[i]} allocated on route with cost {path_cost_G} and min cost {path_min_cost_G}")
    return tot_misallocated_traffic


############MAIN############
# Create an ArgumentParser object
parser = argparse.ArgumentParser(
    description="This program computes the optimal solution to the continuous traffic assignment problem using the Frank-Wolfe algorithm. During the execution of FW, the paths encountered are recorded, together with the flow assigned to them. After running FW, the paths are converted to integer paths. The output of the program is the list of integer paths, the flow on each edge, and some information about the final state of the algorithm. THis variant of the algorithm is optimized for high number of paths"
)

# Add arguments to the parser
parser.add_argument("-N", "--num_nodes", type=int, help="number of nodes in the graph")
parser.add_argument(
    "-edges",
    "--edge_list_file",
    type=str,
    help="path to the txt input file containing the adjacency list",
)
parser.add_argument(
    "-pairs",
    "--sources_destinations",
    type=str,
    help="path to the json input file containing the sources and destinations",
)
parser.add_argument("--quiet", action="store_true", help="Suppress all outputs")
parser.add_argument(
    "-paths",
    "--output_filename_flow_paths",
    type=str,
    default="output_flow_paths.json",
    help="path to the output file containing the paths",
)

parser.add_argument(
    "-state",
    "--output_filename_final_state",
    type=str,
    default="",
    help="path to the output file containing information about the final state of the algorithm.",
)
parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    default=1,
    help="gamma parameter in the cost function. For gamma<1, set step_size_exponent=1 in order to get best convergence performance",
)
parser.add_argument(
    "-t", "--tmax", type=int, default=100, help="max number of iterations"
)
parser.add_argument(
    "-lr_exp",
    "--step_size_exponent",
    type=float,
    default=-1,
    help="the step size for the optimizer at step t will be (t+1)**{-step_size_exponent}. The default is to define tswitch=0.9*tmax and use a step size decreasing as 1/(t+1)**(0.5+0.5*t/tmax). The default is obtained by setting lr_exp to a negative value",
)
parser.add_argument(
    "-rtol",
    "--relative_tolerance_convergence",
    type=float,
    default=1e-6,
    help="relative tolerance in the total energy: when |H(t)-H(t-1)|/H(t-1)<rtol*lr[t], then the algorithm halts. H(t) here is the value of the energy after t iterations, and lr[t] the step size after t iterations.",
)
parser.add_argument(
    "-opt_gap_conv",
    "--relative_optimal_gap_convergence",
    type=float,
    default=0,
    help="relative tolerance in the optimal gap (i.e. difference between the current energy and the lower bound to minimum energy): when |opt_gap(t)|/H(t)<opt_gap_conv, then the algorithm halts. opt_gap(t),H(t) are respectively the value of the optimal gap and the energy after t iterations. WARNING: always set opt_gap_conv=0 when phi is not concave",
)
parser.add_argument(
    "-id",
    "--identifier",
    type=str,
    default="",
    help="run identifier, used to name the output files.",
)
parser.add_argument(
    "--no_output_flow_paths",
     action='store_true',
    help="If this flag is given, the algorithm will not output the final edge flows and the final path flows",
)


# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
N = args.num_nodes
edge_list_filename = args.edge_list_file
sources_destinations_filename = args.sources_destinations
quiet = args.quiet
output_filename_flow_paths = args.output_filename_flow_paths
output_filename_final_state = args.output_filename_final_state
gamma = args.gamma
tmax = args.tmax
lr_exp = args.step_size_exponent
rtol = args.relative_tolerance_convergence
rel_opt_gap_thr = args.relative_optimal_gap_convergence
identifier=args.identifier
no_output_flow_paths=args.no_output_flow_paths

if output_filename_final_state=="":
    output_filename_final_state="./results/state_"+identifier+".txt"

if gamma < 1 and rel_opt_gap_thr > 0:
    print(
        "WARNING: optimal gap convergence criterion is not guaranteed to work for gamma<1. Setting it to zero."
    )
    rel_opt_gap_thr = 0

if not quiet:
    print("Initializing input and output files...")


# reading input files
if not quiet:
    print("Reading input graph...")
G = nk.Graph(n=N, weighted=True, directed=False, edgesIndexed=False)

num_edges=0
with open(edge_list_filename, "r") as edge_list_file:
    for line in edge_list_file:
        u, v= map(int, line.split()[:2])
        G.addEdge(u - 1, v - 1)  # in the input file the nodes start from 1.
        G.setWeight(u - 1, v - 1, 1)
        num_edges+=1

avg_degree=2*num_edges/N 

if not quiet:
    print("Reading sources and destinations...")
with open(sources_destinations_filename, "r") as sources_destinations_file:
    bushes=json.load(sources_destinations_file) #keep in mind the -1 thing!

#converting the json keys which are read as strings, into integers
bushes= {int(key): value for key, value in bushes.items()}
for s in bushes:
    bushes[s]= {int(key): value for key, value in bushes[s].items()}

M=0
for s in bushes:
    for e in bushes[s]:
        M+=bushes[s][e]

if not quiet:
    print("Number of paths: ", M)

# running the optimizer
if not quiet:
    print("Running optimizer...")

if lr_exp >= 0:
    lr = [(1 + t) ** (-lr_exp) for t in range(tmax)]  # step size for the optimizer at step t
else:
    tswitch=int(0.9*tmax)
    lr = [1 / t ** (0.5+ 0.5 * t / tswitch) for t in range(1, tswitch)]+[(1/t)**1.35 for t in range(tswitch,tmax+1)] #empirically this seems to be the best scaling, beating also the line search

start_time=time.time()
(G_traf,G,dict_paths,energies,opt_gaps,flag_conv,final_t) = FW_traffic_assignment_single_commodity(bushes,G,lr,tmax,phi=lambda x: phi(x, gamma),deriv_phi=lambda x: deriv_phi(x, gamma),rtol=rtol,rel_opt_gap_thr=rel_opt_gap_thr)  # compute the flow of each commodity
paths_dict = flows_to_paths(G_traf, dict_paths, bushes, flow_eps=FLOW_EPS)  # compute continuous paths from flows (the flow of a single commuter can be split among several paths each carrying a fraction of the total flow)
integer_paths_dict=continuous_to_max_traf_integer_paths(paths_dict, bushes)# compute integer paths from continuous paths. For each start-end pair, the algorithm puts all the traffic on the path that is carrying the most traffic. For example if a traffic of 1 is split among three paths in the following proportions :0.45,0.35,0.2, the algorithm will choose the first path (carrying 0.45) as integer path.
elapsed_time=time.time()-start_time
#OK
print("FW terminated, computing statistics...")

avg_traf_integer_paths=0
frac_integer_paths_list=[0,0,0,0,0] #fraction of paths returned by the integer projection that were already integer in the continuous solution
tol_integer_paths=[1e-2,1e-3,1e-4,1e-6,0]#level of tolerance in the integrality, e.g., if a path brings a flow of 0.9997 it is considered integer at tolerance 1e-3 but not 1e-4. We look at several levels of tolerance.
frac_shortest_paths = 0 #fraction of the integer paths that are also shortest paths
frac_FW_flow_sp=0
avg_len_shortest_paths=0
shortest_path_energy=0
num_start_end_pairs = 0
avg_len_integer_paths=0 #average length of the integer paths
G_sp = copy_graph(G_traf, reset_weights=True, weight_value=1)
G_sp_traf = copy_graph(G_traf, reset_weights=True, weight_value=0) #graph that will store the traffic (flow) whe the paths are the shortest paths in the topological distance on the graph
G_traf_int = copy_graph(G_traf, reset_weights=True, weight_value=0)
len_shortest_paths_dict={} #takes as keys the pair of nodes (s,e), the value is the node length of the shortest path from s to e in the topological distance.
for s in integer_paths_dict:
    dijkstra = nk.distance.Dijkstra(G_sp, source=s)
    dijkstra.run()
    for e in integer_paths_dict[s]:
        lengths_integer_paths_se=np.array([len(x) for x in integer_paths_dict[s][e][0]])
        lengths_continuous_paths_se=np.array([len(x) for x in paths_dict[s,e][0]])
        num_start_end_pairs+=1
        cont_path_flows=np.array(integer_paths_dict[s][e][2])
        avg_traf_integer_paths+=sum(cont_path_flows)
        avg_len_integer_paths+=sum([(len(p)-1)*integer_paths_dict[s][e][1][i] for i,p in enumerate(integer_paths_dict[s][e][0])])
        zero_tol_int_traf=np.sum(np.floor(cont_path_flows))
        fractional_path_flow=cont_path_flows%1
        for i,tol in enumerate(tol_integer_paths):
            frac_integer_paths_list[i]+=zero_tol_int_traf+np.sum(np.isclose(fractional_path_flow,np.ones_like(fractional_path_flow).astype(int),rtol=0,atol=tol))
        shortest_path_se=dijkstra.getPath(e)
        len_shortest_paths_dict[s,e]=len(shortest_path_se)
        avg_len_shortest_paths+=bushes[s][e]*(len(shortest_path_se)-1)

        idx_integer_shortest_paths=np.where(lengths_integer_paths_se == len(shortest_path_se))[0] #BUG here, returns zero
        for idx in idx_integer_shortest_paths:
            frac_shortest_paths+=integer_paths_dict[s][e][1][idx]
        idx_continuous_shortest_paths=np.where(lengths_continuous_paths_se == len(shortest_path_se))[0]
        for idx in idx_continuous_shortest_paths:
            frac_FW_flow_sp+=paths_dict[s,e][1][idx]

        for k in range(len(shortest_path_se)-1):
            G_sp_traf.setWeight(shortest_path_se[k],shortest_path_se[k+1],G_sp_traf.weight(shortest_path_se[k],shortest_path_se[k+1])+bushes[s][e]) #when using the shortest paths algorithm for routing, we assume that all flow between x and y gets routed through the same shortest path, even if multiple such shortest paths are present.
        for i,path in enumerate(integer_paths_dict[s][e][0]):
            for k in range(len(path) - 1):
                G_traf_int.setWeight(path[k], path[k + 1], G_traf_int.weight(path[k], path[k + 1]) + integer_paths_dict[s][e][1][i])


avg_traf_integer_paths=avg_traf_integer_paths/M  # fraction of total traffic that is routed onto the paths being selected in the integer solution
avg_len_integer_paths=avg_len_integer_paths/M
for i in range(len(frac_integer_paths_list)):
    frac_integer_paths_list[i]=frac_integer_paths_list[i]/M
frac_shortest_paths=frac_shortest_paths/M
frac_FW_flow_sp=frac_FW_flow_sp/M 
avg_len_shortest_paths=avg_len_shortest_paths/M
shortest_paths_energy=sum([phi(w,gamma) for _,_,w in G_sp_traf.iterEdgesWeights()])
integer_paths_energy = sum([phi(w, gamma) for _, _, w in G_traf_int.iterEdgesWeights()])

misallocated_traffic_levels = [misallocated_traffic_optimum(G, paths_dict, rtol=10 ** (-k), atol=0, quiet=True) for k in range(1, 6)]  # checking compatibility of the noninteger paths with the shortest paths with respect to G. This is done with varying degrees of tolerance.
# each entry in misallocated_traffic_levels contains the total traffic allocated on paths that do not satisfy the following condition: cost(path)-min_cost(start,end)<=rtol*min_cost(start,end)+atol. start,end are the endpoints of path. rtol is varied from 1e-1 to 1e-5. atol is set to zero.
 

if not no_output_flow_paths:
    if not quiet:
        print("Writing output paths and flows...")
    with open(output_filename_flow_paths, "a") as output_file_fp:
        json.dump(output_file_fp,integer_paths_dict)


if not quiet:
    print("computing relevant statistics...")


rgap=RGAP(G,G_traf,bushes)
pattern = re.compile(r'adjacency_graph=(\d+)')
matches = pattern.findall(edge_list_filename)
seed = int(matches[0]) #this is the seed used to generate the 
avg_edge_traf_SP=avg_len_shortest_paths*M/num_edges #everage traffic per edge in the case where the paths are chosen to be the shortest paths
avg_edge_traf_FW=sum([w for _,_,w in G_traf.iterEdgesWeights()])/num_edges
avg_edge_traf_FW_int=avg_len_integer_paths*M/num_edges
avg_degen_od_FW= sum([len(paths_dict[key]) for key in paths_dict])/num_start_end_pairs #this computes the average number of paths per origin destination pair (not to be confused with start-end pairs) in the Frank Wolfe solution. 

avg_degen_od_sup_paths=[]
frac_shortest_support_paths=[]
for rtol in [10**(-x) for x in range(2,7)]:
    support_paths_dict=support_paths(G,bushes,atol=0, rtol=rtol)
   
    tmp_frac_shortest_support_paths=0
    for s,e in support_paths_dict:
        len_sup_paths_se=np.array([len(x) for x in support_paths_dict[s,e]])
        tmp_frac_shortest_support_paths+=np.mean(len_sup_paths_se==len_shortest_paths_dict[s,e])
    frac_shortest_support_paths.append(tmp_frac_shortest_support_paths/num_start_end_pairs)
    avg_degen_od_sup_paths.append(np.mean(np.array([len(sup_paths) for sup_paths in support_paths_dict.values()])))
   


header="N M d gamma tmax seed FW_fin_energy FW_fin_opt_gap int_FW_path_energy shortest_path_energy frac_shortest_paths avg_traf_int_paths avg_len_int_paths time avg_len_shortest_path avg_traf_shortest_paths avg_traf_FW avg_traf_int_FW RGAP avg_degen_od_FW identifier num_start_end_pairs "+" ".join([f"misall_traffic_1e-{x}" for x in range(1,6)])+" "+" ".join([f"sup_paths_od_degen_rtol_1e-{x}" for x in range(2,7)])+" "+" ".join([f"frac_integer_paths_tol_{x:1.0e}" for x in tol_integer_paths])+" frac_FW_flow_SP"+" "+" ".join([f"frac_sup_shortest_paths_tol_1e-{x}" for x in range(2,7)])+" final_t init_energy"+"\n"
if not quiet:
    print("writing final results on file...")
output_file_final_state=open(output_filename_final_state, "a")

if is_file_empty(output_filename_final_state):
   output_file_final_state.write(header) 


output_string=f"{N} {M} {avg_degree} {gamma} {tmax} {seed} {energies[-1]:.11e} {opt_gaps[-1]:.4e} {integer_paths_energy:.11e} {shortest_paths_energy :.6e} {frac_shortest_paths:.4e} {avg_traf_integer_paths:.5e} {avg_len_integer_paths:.5e} {elapsed_time:.4e} {avg_len_shortest_paths:.4e} {avg_edge_traf_SP:.4e} {avg_edge_traf_FW:.4e} {avg_edge_traf_FW_int:.4e} {rgap:.4e} {avg_degen_od_FW:.4e} {identifier} {num_start_end_pairs} "+" ".join(map(str, misallocated_traffic_levels))+" "+" ".join(map(str, avg_degen_od_sup_paths))+" "+" ".join(map(str, frac_integer_paths_list))+f" {frac_FW_flow_sp:1.5e}"+" "+" ".join(map(str, frac_shortest_support_paths))+f" {final_t} {energies[0]:.11e}"+"\n"
output_file_final_state.write(output_string)
output_file_final_state.close()
if not quiet:
    print("program excecuted successfuly. Bye bye!")
    # fraction of integer paths that have length equal to that of the shortest path between the same endpoints
    # fraction of paths for which FW converges to an integer path (meaning the path found by Frank-Wolfe carries a traffic of 1)
    # average traffic carried by the paths with maximum traffic. This is a measure of how 'noninteger' the solution is. If for a certain path the traffic is 1, then Frank-Wolfe already outputted an integer path.
