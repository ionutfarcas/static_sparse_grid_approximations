from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *
from sg_lib.operation.interpolation_to_spectral import *

from config.config import *
    
if __name__ == '__main__':

	## target-function
	f_ref = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + 2*x[2] * x[3] + 3
	
	## sparse grid setup ##
	#######################
	Grid_obj 				= Grid(dim, level, level_to_nodes, left_bounds, right_bounds, weights)	
	Multiindex_obj 			= Multiindex(dim)
	InterpToSpectral_obj 	= InterpolationToSpectral(dim, level_to_nodes, left_bounds, right_bounds, weights,level, Grid_obj)

	## sparse grid multi-index set 
	multiindex_set = Multiindex_obj.get_std_total_degree_mindex(level)
	#######################

	# evaluate target function at sparse grid points
	for m, multiindex in enumerate(multiindex_set):

		mindex_grid_inputs = Grid_obj.get_sg_surplus_points_multiindex(multiindex)
		
		mapped_sg_point = np.zeros(dim)
		for sg_point in mindex_grid_inputs:
			mapped_sg_point[0] = left_stoch_boundary[0] + (right_stoch_boundary[0] - left_stoch_boundary[0])*sg_point[0]
			mapped_sg_point[1] = left_stoch_boundary[1] + (right_stoch_boundary[1] - left_stoch_boundary[1])*sg_point[1]
			mapped_sg_point[2] = left_stoch_boundary[2] + (right_stoch_boundary[2] - left_stoch_boundary[2])*sg_point[2]
			mapped_sg_point[3] = left_stoch_boundary[3] + (right_stoch_boundary[3] - left_stoch_boundary[3])*sg_point[3]

			y_eval = f_ref(mapped_sg_point)

			InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, y_eval)

		InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, Grid_obj)

	# after all evaluations are done, we obtain the sparse grid interpolant
	f_interp = lambda x: InterpToSpectral_obj.eval_operation_sg(multiindex_set, x)

	test_points 		= np.random.uniform(0, 1, size=(10, dim))
	mapped_test_points 	= Grid_obj.map_std_sg_surplus_points(test_points, left_stoch_boundary, right_stoch_boundary)

	# for the considered example, the interpolant should be exact for test points within bounds!
	y1 = np.array([f_ref(x) for x in mapped_test_points])
	y2 = np.array([f_interp(x) for x in test_points])


	print("\033[1m SUROGATE MODEL PART \033[0m")
	print("\033[1m reference evaluations \033[0m")
	print(y1)

	print("\033[1m SG interpolant evaluations \033[0m")
	print(y2)

	print("\033[1m pointwise differences \033[0m")
	print(y1 - y2)
	print("\033[1m ******************* \033[0m")



	print("\033[1m UQ PART \033[0m")
	# get equivalent sparse grid spectral coefficients and basis
	coeff_SG, basis_SG = InterpToSpectral_obj.get_spectral_coeff_sg(multiindex_set)

	np.save('results/coeff_SG.npy', coeff_SG)
	np.save('results/basis_SG.npy', basis_SG)


	# get statistics
	mean_est 		= InterpToSpectral_obj.get_mean(coeff_SG)
	var_est 		= InterpToSpectral_obj.get_variance(coeff_SG)

	print("\033[1m mean and variance \033[0m")
	print(mean_est, var_est)

	# get all Sobol indices for sensitivity analysis
	multiindex_bin 		= Multiindex_obj.get_poly_mindex_binary(dim)
	all_Sobol_indices 	= InterpToSpectral_obj.get_all_sobol_indices(multiindex_bin, coeff_SG, multiindex_set)

	print("\033[1m Sobol indices for sensitivity analysis \033[0m")
	print(all_Sobol_indices)
	print("\033[1m ******************* \033[0m")