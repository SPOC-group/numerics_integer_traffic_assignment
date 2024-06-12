import networkit as nk
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import sys

gamma = 2 # non-linearity exponent
degree = 3 # degree of the graph


# Function that computes the shortest paths : 
def Routing_Greedy_procedure(G,M,N,start_nodes,end_nodes,gamma,tmax):
    paths = []

    for u,v in G.iterEdges():
        G.setWeight(u,v,1) # initialize all weights to zero

    G_traf=copy.deepcopy(G) 
    for u,v in G_traf.iterEdges():
        G_traf.setWeight(u,v,0) #initialize all weights to zero (traffic is at zero in the beginning)

    # Compute each path individually 
    for nu in range(M):
        dijkstra = nk.distance.Dijkstra(G, source=start_nodes[nu],target=end_nodes[nu])
        dijkstra.run()
        path = dijkstra.getPath(end_nodes[nu])
        paths.append(path)   

        # after getting the path, add its contribution :
        for i in range(len(path)-1):
            G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])+1)
        
    for u,v in G.iterEdges():
        G.setWeight( u,v,  phi(((G_traf.weight(u,v)+1)  ), gamma)- phi(((G_traf.weight(u,v)) ), gamma)) 


    # Iterating 
    flag_conv=False
    iteration_count = 0
    for _ in range(tmax):
        if flag_conv:
            break
        flag_conv=True
        shuffled_indices = list(range(M))
        random.shuffle(shuffled_indices)
        for nu in shuffled_indices : #loop over paths to update them
            path=paths[nu]
        
            # I first remove the contribution of path nu : 
            for i in range(len(path)-1): 
                G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])-1)   # to get the traffic without path nu
                G.setWeight(path[i],path[i+1],phi((G_traf.weight(path[i],path[i+1])+1) , gamma )- phi((G_traf.weight(path[i],path[i+1])) , gamma ))  # I keep the +1 in the beginning because it has already been substracted in previous line 
            
            # Compute the new optimal path nu
            dijkstra = nk.distance.Dijkstra(G, source=start_nodes[nu],target=end_nodes[nu]) 
            dijkstra.run()
            new_path=dijkstra.getPath(end_nodes[nu])
                        
            if(new_path!=path):# check for convergence (it's enough to get 1 false to continue in time)
                flag_conv=False 
            
            paths[nu]=new_path
            for i in range(len(new_path)-1): #add the contribution of the new path nu
                G_traf.setWeight(new_path[i],new_path[i+1],G_traf.weight(new_path[i],new_path[i+1])+1) # i am already adding 1 where the path passes 
                G.setWeight(new_path[i],new_path[i+1], phi((G_traf.weight(new_path[i],new_path[i+1])+1),gamma)- phi((G_traf.weight(new_path[i],new_path[i+1])),gamma)) # i again add a one for the next nu to be selected
        iteration_count = iteration_count+1 

    G_energy=copy.deepcopy(G) 
    for u,v in G_energy.iterEdges():
        G_energy.setWeight( u,v,phi(((G_traf.weight(u,v))  ), gamma)) 

    return G_traf, G_energy, G, paths,iteration_count 
# End of the function that computes the shortest paths

# Following lines are just useful to retreive, from the adjacency file, the number of nodes in the graph:
N = float('-inf')
# Read the file and find the maximum value in the first and second columns
with open('input/Adjacency_list.dat', 'r') as file:
    for line in file:
        line = line.strip().split()
        if len(line) >= 2:
            edge_start = int(line[0])
            edge_end = int(line[1])
            if edge_start > N:
                N = edge_start
            if edge_end > N:
                N = edge_end
# import the graph :  ---------------------------------------------------
G = nk.Graph()
for i in range(N):
    G.addNode()

# Read the adjacency list from the file
with open('input/Adjacency_list.dat', 'r') as file:
    for line in file:
        line = line.strip().split()
        edge_start = int(line[0])-1
        edge_end = int(line[1])-1
        G.addEdge(edge_start, edge_end)
G = nk.graphtools.toWeighted(G)

# import the sources,destinations :  ---------------------------------------------------
start_nodes = []
end_nodes = []
M = 0 
with open('input/OD_pairs.dat', 'r') as file:
    for line in file:
        line = line.strip().split()
        start = int(line[0])
        end = int(line[1])
        start_nodes.append(start-1)
        end_nodes.append(end-1)
        M = M + 1 # couting the number of paths.


# uselful functions :  ---------------------------------------------------
def phi(x,gamma):
    return x**(gamma)

def Total_energy(G_energy):
    total = 0
    for u,v in G_energy.iterEdges():
        total = total + G_energy.weight(u,v)
    return total 

def Average_length(paths,M): 
    total_length = 0 
    for path in paths : 
        total_length = total_length + (len(path)-1)
    return total_length/M


# apply the routing function : ---------------------------------------------------
G_traf_GP, G_energy_GP, G_SP, paths_GP, t_final = Routing_Greedy_procedure(G,M,N,start_nodes,end_nodes,gamma,tmax=10^15)

# Store the additional data in variables (energy of the configuration and average length of paths): -------------------
# Total energy : 
total_energy_GP = Total_energy(G_energy_GP)
# Average length of the path : 
average_length_GP = Average_length(paths_GP, M)

# Print the parameters : ------------------------------------------------
print("Greedy algorithm - results : ")
print("Configuration energy :", total_energy_GP) 
print("Average length :",average_length_GP)

# Record the paths : ------------------------------------------------
with open('output/Greedy_paths.dat', 'w') as file:
    for path in paths_GP:
        shifted_path = [node + 1 for node in path]
        file.write(' -> '.join(map(str, shifted_path)) + '\n')



