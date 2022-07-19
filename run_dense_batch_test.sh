module load cuda/10.2
# source /is/cluster/wliu/.bashrc
export PATH="/is/cluster/wliu/anaconda3/envs/pt2/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH"
echo $PATH
. "/is/cluster/wliu/anaconda3/etc/profile.d/conda.sh"
conda activate pt2
python run_dense_batch.py --order_ind $1 --config /is/cluster/wliu/unsup3d_latent_new/experiments/test_synface_additive.yml --gpu 0 --num_workers 4 --seed 0
python run_dense_batch.py --order_ind $1 --config /is/cluster/wliu/unsup3d_latent_new/experiments/test_synface_additive.yml --gpu 0 --num_workers 4 --seed 1
python run_dense_batch.py --order_ind $1 --config /is/cluster/wliu/unsup3d_latent_new/experiments/test_synface_additive.yml --gpu 0 --num_workers 4 --seed 2
python run_dense_batch.py --order_ind $1 --config /is/cluster/wliu/unsup3d_latent_new/experiments/test_synface_additive.yml --gpu 0 --num_workers 4 --seed 3
