Various versions of RITAP were used to solve ITAP. They are all based on the FW algorithm but they differ with respect to the graph being directed or not, the form of the activation functions.

'frank_wolfe_optimizer_paths_v2highM.py' is a version of RITAP optimized for very high M. In runs on undirected graphs with uniform (i.e. the same on all edges) activation functions
run with 
python3  frank_wolfe_optimizer_paths_v2highM.py -N 32 -edges ./input/adjacency_graph=0_M=205.dat -pairs ./input/desPairs_0_M=205.json --gamma 2 --tmax 1000 -lr_exp -1  -rtol 1e-9 -opt_gap_conv 1e-9 --no_output_flow_paths --identifier test_run


'frank_wolfe_optimizer_paths_v2highM_directed.py' is the same as the previous code, but it runs on directed graphs. Basically the traffic on one edge is counted separately based on the direction. It takes as input a the adjacency list of directed graph.
run with
python3  frank_wolfe_optimizer_paths_v2highM_directed.py -N 32 -edges ./input/adjacency_graph=0_M=371.dat -pairs ./input/desPairs_0_M=371.json --gamma 2 --tmax 1000 -lr_exp -1  -rtol 1e-9 -opt_gap_conv 1e-9 --no_output_flow_paths --identifier test_run


'frank_wolfe_optimizer_paths_v2highM_directed_BPR.py' runs on directed graphs and works with BPR type nonlinearities. The parameters of the BPR function can depend on the edge and are read from the file containing the network adjacency list.

run with 
python3 frank_wolfe_optimizer_paths_v2highM_directed_BPR.py --network_file "./input/EMA_net.tntp" --sources_destinations "./input/EMA_trips.json"  --tmax 1000 -lr_exp -1  -rtol 0 -opt_gap_conv 0 --no_output_flow_paths --identifier 'EMA_test' >print_out.txt


'frank_wolfe_optimizer_paths_v2highM_alternative_degen.py' employs a different method to compute the degeneracy. It basically counts the paths with nonzero flow. Using this more crude method is necessary when computing the true degeneracy is too computationally expensive  

run with 
python3  frank_wolfe_optimizer_paths_v2highM_directed_alternative_degen.py -N 32 -edges ./input/adjacency_graph=0_M=371.dat -pairs ./input/desPairs_0_M=371.json --gamma 2 --tmax 1000 -lr_exp -1  -rtol 1e-9 -opt_gap_conv 1e-9 --no_output_flow_paths --identifier test_run
