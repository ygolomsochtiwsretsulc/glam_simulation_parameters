# glam_simulation_parameters
creates the intial grid of cosmological parameters for eROSITA GLAM simulations

Please follow this procedure
 * add a few lines in the readme about your proposition 
 * create a folder where you host the code used (and instruction to execute it) to generate a list of cosmological parameters to sample as well as the list


## Proposition 1

J. Comparat's proposition is in folder example_grid_w_pydoe

The list of proposed cosmological parameters is in the file: cosmo_params.ascii

It is generated with the following commands
```
conda activate eroconda
pip install pyDOE
cd glam_simulation_parameters/example_grid_w_pydoe
python glam_lhs.py 
```
