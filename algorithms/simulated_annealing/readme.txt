To run the simulated annealing execute the following command in the terminal

python simulated_annealing.py -N 100 -edges ./test_inputs/adjacency_graph=0_M=100.dat -pairs ./test_inputs/desPairs_0_M=100.dat --gamma 2 -t_mcmc 2 --beta_min 0.1 --annealing_steps 100 --greedy_steps 30