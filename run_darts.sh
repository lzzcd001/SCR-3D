#!/bin/bash

module load cuda/10.2
conda activate pt2

case $1 in
	0)
		python run_darts.py --config experiments/train_synface_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total
		;;

	1)
		python run_darts.py --config experiments/train_syncar_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total
		;;

	2)
		python run_darts.py --config experiments/train_cat_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total
		;;

	3)
		python run_darts.py --config experiments/train_celeba_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total
		;;

	4)
		python run_darts.py --config experiments/train_synface_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total --inner_steps 1
		;;

	5)
		python run_darts.py --config experiments/train_syncar_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total --inner_steps 1
		;;

	6)
		python run_darts.py --config experiments/train_cat_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total --inner_steps 1
		;;

	7)
		python run_darts.py --config experiments/train_celeba_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --lam_adj 1e6 --darts_obj loss_total --inner_steps 1
		;;	
esac	


