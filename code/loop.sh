#!/bin/bash


#random_chance_vals	= [0.5,0.1,0.05,0.01,0.001]
#discount_vals 		= [0.999, 0.99, 0.90, 0.80,0.50]
#learning_rate_vals	= [0.4,0.2,0,1,0.05,0.01]
#sigma_vals			= [20,10,5,1,0.1]
#sd_vals				= [5,2,1,0.5,0.1]
#action_hidden_layers= [500,200,100,50,10]
#value_hidden_layers	= [500,200,100,50,10]

random_chance_vals=(0.1)
discount_vals=(0.99)
learning_rate_vals=(0.01 0.02)
sigma_vals=(10)
sd_vals=(1)
action_hidden_layers=(200)
value_hidden_layers=(200)

for var in "${random_chance_vals[@]}"
do
	for disc_val in "${discount_vals[@]}"
	do
		for lr_val in "${learning_rate_vals[@]}"
		do
			for sig_val in "${sigma_vals[@]}"
			do
				for sd_val in "${sd_vals[@]}"
				do
					for ah_val in "${action_hidden_layers[@]}"
					do
						for vh_val in "${value_hidden_layers[@]}"
						do
							python main.py $var $disc_val $lr_val $sig_val $sd_val $ah_val $vh_val
						done
					done
				done
			done
		done
	done
done