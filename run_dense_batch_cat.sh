module load cuda/10.2
conda activate pt2
python run_dense_batch.py --order_ind $1 --config experiments/train_cat_additive.yml --gpu 0 --num_workers 4 --seed 0
