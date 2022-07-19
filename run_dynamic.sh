#!/bin/bash

module load cuda/10.2
conda activate pt2

case $1 in
	0)
		python run_dynamic.py --config experiments/train_synface_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dynamic_emb cos --dense_order 3 2 1 0
		;;

	1)
		python run_dynamic.py --config experiments/train_synface_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb dense
		;;
	
	2)
		python run_dynamic.py --config experiments/train_synface_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb attention
		;;

	3)
		python run_dynamic.py --config experiments/train_syncar_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb cos
		;;
	
	4)
		python run_dynamic.py --config experiments/train_syncar_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb dense
		;;

	5)
		python run_dynamic.py --config experiments/train_syncar_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb attention
		;;

	6)
		python run_dynamic.py --config experiments/train_cat_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb cos
		;;

	7)
		python run_dynamic.py --config experiments/train_cat_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb dense
		;;

	8)
		python run_dynamic.py --config experiments/train_cat_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb attention
		;;
	
	9)
		python run_dynamic.py --config experiments/train_celeba_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --dense_order 3 2 1 0 --dynamic_emb attention
		;;

	10)
		python run_darts.py --config experiments/train_synface_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --non_darts --dense_order 3 2 1 0
		;;

	11)
		python run_darts.py --config experiments/train_syncar_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --non_darts --dense_order 3 2 1 0
		;;

	12)
		python run_darts.py --config experiments/train_cat_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --non_darts --dense_order 3 2 1 0
		;;

	13)
		python run_darts.py --config experiments/train_celeba_additive.yml --gpu 0 --num_workers 4 --seed 0 --no_dag_loss --non_darts --dense_order 3 2 1 0
		;;
esac	


