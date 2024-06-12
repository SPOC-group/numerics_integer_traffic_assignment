import numpy as np
import argparse
import networkit as nk
import numpy as np
import networkx as nx   

INF_BETA=1e16 #value of beta corresponding to zero temperature

def phi(x,gamma):
    return x**gamma


def shortest_path_optimizer(G,start_nodes,end_nodes):
    #compute shortest paths (used as baseline)
    shortest_paths=[]
    # Initialize each path using Dijkstra
    for nu in range(M):
        dijkstra = nk.distance.Dijkstra(G, source=start_nodes[nu],target=end_nodes[nu])
        dijkstra.run()
        path = dijkstra.getPath(end_nodes[nu])
        shortest_paths.append(path.copy()) 
    return shortest_paths


def energy(path,G):
    """
    G here is a weighted graph. this function returns the sum of the weights of G along the path.
    """
    
    energy=0
    for k in range(len(path)-1):
        energy+=G.weight(path[k],path[k+1])
    return energy


def global_energy(G,paths,phi):
    N=G.numberOfNodes()
    G_traf=nk.graph.Graph(n=N, weighted=True, directed=False, edgesIndexed=False)
    for u,v in G.iterEdges():
        G_traf.addEdge(u,v)
        G_traf.setWeight(u,v,0) #initialize all weights to zero (traffic is at zero in the beginning)
    # Initialize each path using Dijkstra
    for path in paths:
        # total traffic on the graph
        for i in range(len(path)-1):
            G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])+1)
    
    result=0
    for u,v,w in G_traf.iterEdgesWeights():
        result=result+phi(w)
    return result


def propose_path_mcmc(G,beta,start_node,end_node):
    """
    This function proposes a new path in he metropolis step of the mcmc. The path is built autoregressively.
    G should be a directed weighted graph.
    This function returns the proposed path, together with its log probability under the proposal distribution.
    
    Inside this function, G changes as new nodes are added to the path: every time a new vertex u is added to the path, all the edges v->u for v a neighbour of u are removed.
    Removing all the incoming edges into nodes in the path ensure that the walk is self avoiding.
    When the function is finished executing, the graph G is returned to its original state.
    """
    
    curr_node=start_node
    path=[start_node]
    removed_edges=[]
    #keeping track of the log probability, to be used for metropolis acceptance rate.
    path_log_prob=0
    #loop over path length
    while(curr_node!=end_node):
        #removing edges going into the last node in the path
        edges_to_remove=[]
        for v in G.iterInNeighbors(curr_node): #removing incoming edges
            edges_to_remove.append([v,curr_node, G.weight(v,curr_node)])
        for edge in edges_to_remove:
            G.removeEdge(edge[0],edge[1]) 
        removed_edges=removed_edges+edges_to_remove
        
        out_degree=G.degreeOut(curr_node)
        C_path_y=np.zeros(out_degree) #array containing the C coefficients for the current iteration.
        neigh_curr_node=np.zeros(out_degree, dtype=np.uintc) #neighbours of the current node
        i=0
        for y,w in G.iterNeighborsWeights(curr_node): #iterates over the out-neighbours of curr_node
            spsp = nk.distance.SPSP(G,[y]) #WARNING: if there is no path from y to e, the getDistance function will return 1.797e+308, which is virtually infinity, unless beta=0.
            # therefore if beta=0, the algorithm will not work.
            spsp.run()
            V_e_y=spsp.getDistance(y,end_node) 
            C_path_y[i]=V_e_y+w
            neigh_curr_node[i]=y
            i=i+1
        
        #defining the transition probability
        C_path_y=C_path_y-np.min(C_path_y) #regularizing
        P_trans=np.exp(-beta*C_path_y)   #overflows can be expected here. When no path goes from the neighbour to end_node, the corresponding entry of C_path_y will be 1.797e308
        P_trans=P_trans/np.sum(P_trans) #normalize the probability
        idx_new_node=np.random.choice(a=len(P_trans),size=None, replace=True, p=P_trans)
        
        new_node=neigh_curr_node[idx_new_node].astype(np.uintc)
        path.append(new_node)
        path_log_prob+=np.log(P_trans[idx_new_node])
        curr_node=new_node
    
    #restore removed edges so that G is not modified at the end. This avoids creating copies of G every time.
    for edge in removed_edges:
        G.addEdge(edge[0],edge[1])
        G.setWeight(edge[0],edge[1],edge[2])
            
    return path, path_log_prob


def compute_prop_path_log_prob(G, beta, path, start_node, end_node):
    """
    Function that computes the probability of a path under the proposal distribution.
    path is a python list whose first element must be start_node and whose last element must be end_node
    """
    removed_edges=[]
    if(path[0]!=start_node):
        print("Error: the first element of path must be start_node")
    #keeping track of the log probability, to be used for metropolis acceptance rate.
    path_log_prob=0
    #loop over path length    
    for k in range(len(path)-1):
        curr_node=path[k]
        #removing edges going into the last node in the path
        edges_to_remove=[]
        for v in G.iterInNeighbors(curr_node): #removing incoming edges
            edges_to_remove.append([v,curr_node, G.weight(v,curr_node)])
        for edge in edges_to_remove:
            G.removeEdge(edge[0],edge[1]) 
        removed_edges=removed_edges+edges_to_remove
        out_degree=G.degreeOut(curr_node)
        C_path_y=np.zeros(out_degree) #array containing the C coefficients for the current iteration.
        neigh_curr_node=np.zeros(out_degree, dtype=np.uintc) #neighbours of the current node
        i=0
        #error is after here
        for y,w in G.iterNeighborsWeights(curr_node): #iterates over the out-neighbours of curr_node
            spsp = nk.distance.SPSP(G,[y]) #WARNING: if there is no path from y to e, the getDistance function will return 1.8e+308, which is virtually infinity, unless beta=0.
            # therefore if beta=0, the algorithm will not work.
            spsp.run()
            V_e_y=spsp.getDistance(y,end_node) 
            C_path_y[i]=V_e_y+w
            neigh_curr_node[i]=y
            i=i+1
        #defining the transition probability
        C_path_y=C_path_y-np.min(C_path_y) #regularizing
        log_P_trans=-beta*C_path_y
        #P_trans=P_trans/np.sum(P_trans) #normalize the probability
        new_node=path[k+1]
        idx_new_node=np.where(neigh_curr_node==new_node)[0]
        path_log_prob+=log_P_trans[idx_new_node]-np.log(np.sum(np.exp(log_P_trans)))
        
    #restore removed edges so that G is not modified at the end. This avoids creating copies of G every time.
    for edge in removed_edges:
        G.addEdge(edge[0],edge[1])
        G.setWeight(edge[0],edge[1],edge[2])
    return path_log_prob


def mcmcm_saw(beta,G,start_node,end_node, tmax, init_path=None):
    """
    Function that implements a whole mcmc simulation.
    beta is the inverse temperature used in the simulation.
    G is the weighted and directed graph over which self-avoiding paths are sampled.
    start_node, end_node are repsectively the starting and ending point of the path.
    tmax is the length of the simulation.
    Returns the final path of the simulation
    """
    if(init_path == None):
        #initializing at the shortest path   
        path,_=propose_path_mcmc(G,1e9,start_node,end_node)
    else:
        path=init_path.copy()
    
    path_log_prob=compute_prop_path_log_prob(G, beta, path, start_node, end_node)
    path_energy=energy(path,G)
    num_accepted=0
    log_random_numbers=np.log(1-np.random.uniform(low=0.0, high=1.0, size=tmax)) #this way one does not get log(0).
    for t in range(tmax):
        prop_path, prop_path_log_prob=propose_path_mcmc(G, beta, start_node, end_node) #the function propose_path_mcmc does not alter G
        prop_path_energy=energy(prop_path,G)
        log_p_acc=beta*(path_energy-prop_path_energy)+(path_log_prob-prop_path_log_prob)
        if(log_random_numbers[t]<log_p_acc):
            num_accepted+=1
            path=prop_path.copy()
            path_energy=prop_path_energy
            path_log_prob=prop_path_log_prob
    return path


def annealed_optimizer(G,start_nodes,end_nodes,beta_schedule,phi,mcmc_steps=10,mcmc_seed=None,quiet=True, conv_iters=3):
    """
    Simulated annealing based optimizer for the multiple path problem. 
    start_nodes and end_nodes are two lists containing respectively the source and destination of each path
    beta_schedule is a lsit containing the valeus of beta to use in each iteration.
    phi is the nonlinearity in the hamiltonian
    In each iteration, an mcmc (which draws 'mcmc_steps' samples) is performed for every path: this is the way paths are updated.
    If the energy stays constant for conv_iter iterations, then we say tha algorithm has converged.
    if reinit_path=True, then each an mcmc is started to resample a path, the initial condition of the mcmc will not be the current value of the path. Instead the mcmc will be initialized with the shortest path (w.r.t the weights in Gd) between the source and destination.
    
    """
    M=len(start_nodes)
    N=G.numberOfNodes()
    Gd=nk.graph.Graph(n=N, weighted=True, directed=True, edgesIndexed=False) #this graph is a directed version of G. It should not be subject to modifications throughout the dynamics.
    for u,v in G.iterEdges():
        Gd.addEdge(u,v)
        Gd.setWeight(u,v,1)
        Gd.addEdge(v,u)
        Gd.setWeight(v,u,1)    
    
    paths = []
    G_traf=nk.graph.Graph(n=N, weighted=True, directed=False, edgesIndexed=False)
    for u,v in G.iterEdges():
        G_traf.addEdge(u,v)
        G_traf.setWeight(u,v,0) #initialize all weights to zero (traffic is at zero in the beginning)

    # Initialize each path using Dijkstra
    for nu in range(M):
        dijkstra = nk.distance.Dijkstra(G, source=start_nodes[nu],target=end_nodes[nu])
        dijkstra.run()
        path = dijkstra.getPath(end_nodes[nu])
        paths.append(path.copy())   

        # total traffic on the graph
        for i in range(len(path)-1):
            G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])+1)

    for u,v in Gd.iterEdges(): #initializing cost on all the graph: this way one will only have to modify the cost on the edges traversed by the considered path
        Gd.setWeight( u,v,   phi((G_traf.weight(u,v)+1))- phi(G_traf.weight(u,v)))
    total_energy=sum([phi(w) for _,_,w in G_traf.iterEdgesWeights()]) #this is the objective functino to minimize. When it becomes stationary, the algorithm stops. 
    count_conv=0
    np.random.seed(mcmc_seed)
    flag_conv=False
    for t,beta in enumerate(beta_schedule):
        if(not quiet):
            print(f"t={t}, beta={beta:.2e}, energy={total_energy:.3e}")

        for nu in range(M):
            path=paths[nu]
            # Remove the contribution of path nu
            for i in range(len(path)-1): 
                G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])-1) # to get the traffic without path nu
                Gd.setWeight(path[i],path[i+1],phi((G_traf.weight(path[i],path[i+1])+1))- phi((G_traf.weight(path[i],path[i+1]))))
                Gd.setWeight(path[i+1],path[i],phi((G_traf.weight(path[i],path[i+1])+1))- phi((G_traf.weight(path[i],path[i+1]))))
            #Gd now contains the potential seen by path nu 
            new_path=mcmcm_saw(beta, Gd, start_nodes[nu], end_nodes[nu], mcmc_steps, path) #compute update path using the mcmc. If None is passed as an argument instead of 'path', then the mcmc uses tha shortest path as initialization.
            paths[nu]=new_path.copy()
            for i in range(len(new_path)-1): #add the contribution of the new path nu
                G_traf.setWeight(new_path[i],new_path[i+1],G_traf.weight(new_path[i],new_path[i+1])+1) #re adding the contribution of path nu
                Gd.setWeight(new_path[i],new_path[i+1], phi((G_traf.weight(new_path[i],new_path[i+1])+1))- phi((G_traf.weight(new_path[i],new_path[i+1])))) 
                Gd.setWeight(new_path[i+1],new_path[i],  phi((G_traf.weight(new_path[i],new_path[i+1])+1))- phi((G_traf.weight(new_path[i],new_path[i+1])))) 
                
        #check for convergence in energy
        new_total_energy=sum([phi(w) for _,_,w in G_traf.iterEdgesWeights()])

        if(new_total_energy==total_energy):
            count_conv+=1
            if(count_conv>=conv_iters):
                flag_conv=True
                break
        else: 
            count_conv=0


        total_energy=new_total_energy
    return paths,flag_conv


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Command-line parser for the multipath optimizer.")

# Add arguments to the parser
parser.add_argument('-N','--num_nodes', type=int, help='number of nodes in the graph')
parser.add_argument('-edges','--edge_list_file', type=str, help='path to the input file containing the adjacency list')
parser.add_argument('-pairs','--sources_destinations', type=str, help='path to the input file containing the sources and destinations')
parser.add_argument('--quiet', action='store_true', help='Suppress all outputs')
parser.add_argument('-paths','--output_file_paths', type=str, default="output_paths.txt", help='path to the output file containing the paths')
parser.add_argument('-flow', '--output_file_flow', type=str, default="output_flow.txt", help='path to the output file containing the flow')
parser.add_argument('-g', '--gamma', type=float, default=1, help='gamma parameter in the cost function')
parser.add_argument('-t_mcmc', '--mcmc_steps', type=int, default=1, help='number of mcmc steps in each mcmc run')
parser.add_argument('-bm', '--beta_min', type=float, default=1, help='minimum value of beta in the simulated annealing')
parser.add_argument('-as','--annealing_steps', type=int, default=10, help='number of annealing (i.e., finite beta) steps in the simulated annealing. The total number of steps is given by annealing_steps+greedy_steps')
parser.add_argument('-gs','--greedy_steps', type=int, default=20, help='number of steps at beta=infinity, after the annealing schedule has terminated. The total number of steps is given by annealing_steps+greedy_steps')
parser.add_argument('-s','--seed', type=int, default=None, help='seed for the MCMC random number generator')


# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
edge_list_filename = args.edge_list_file
sources_destinations_filename = args.sources_destinations
quiet=args.quiet
output_filename_paths=args.output_file_paths
output_filename_flow=args.output_file_flow
gamma=args.gamma
mcmc_steps=args.mcmc_steps
beta_min=args.beta_min
annealing_steps=args.annealing_steps
greedy_steps=args.greedy_steps
mcmc_seed=args.seed

if(not quiet):
    print("Initializing input and output files...")

N=args.num_nodes

#reading input files
if(not quiet):
    print("Reading input graph...")
G=nk.Graph(n=N, weighted=False, directed=False, edgesIndexed=False)

with open(edge_list_filename, 'r') as edge_list_file:
    
    for line in edge_list_file:
        u, v, w = map(int, line.split())
        G.addEdge(u-1, v-1) #in the input file the nodes start from 1.


if(not quiet):
    print("Reading sources and destinations...")
start_nodes=[]
end_nodes=[]
with open(sources_destinations_filename, 'r') as sources_destinations_file:
    for line in sources_destinations_file:
        u, v= map(int, line.split())
        start_nodes.append(u-1)
        end_nodes.append(v-1)

M=len(start_nodes)
if(not quiet):
    print("Number of paths: ",M)

#running the optimizer
if(not quiet):
    print("Running optimizer...")
beta_schedule=beta_min*(annealing_steps)/(annealing_steps-np.arange(0,annealing_steps))
beta_schedule=np.concatenate((beta_schedule,INF_BETA*np.ones(greedy_steps))) #this way the last 'greedy_steps' iterations will be at zero temperature and the algorithm is guaranteed to converge
np.random.seed(mcmc_seed)
paths,flag_conv=annealed_optimizer(G,start_nodes,end_nodes,beta_schedule,phi=lambda x:phi(x,gamma),mcmc_steps=mcmc_steps,quiet=quiet, conv_iters=3)

if(not quiet):
    print("Converged: ",flag_conv)
    print("Writing output paths...")
with open(output_filename_paths,'w') as output_file_paths:
    for i,path in enumerate(paths):
        output_file_paths.write(f"{i+1} "+" ".join(map(str,np.array(path)+1))+"\n")


if(not quiet):
    print("Writing output flow...") 

G_flow=nk.graph.Graph(n=N, weighted=True, directed=False, edgesIndexed=False)
for u,v in G_flow.iterEdges():
    G_flow.setWeight(u,v,0)

for path in paths:
    for i in range(len(path)-1):
        G_flow.setWeight(path[i],path[i+1],G_flow.weight(path[i],path[i+1])+1)

with open(output_filename_flow,'w') as output_file_flow:
    for u,v,w in G_flow.iterEdgesWeights():
        output_file_flow.write(f"{u+1} {v+1} {w}\n")