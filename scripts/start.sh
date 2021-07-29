EXPERIMENT_ID=exp_001

hostname
nvidia-smi
df -h -l
python --version
python -c "import torch; print(torch.__version__)"
pip list

echo '~~~~~~~~~~~~CLONING AND BUILDING CODE~~~~~~~~~~'
mkdir -p /programs/python/lib/python
export PYTHONPATH=/programs/python/lib/python:`pwd`/lib:$PYTHONPATH

cd /programs
git clone https://github.com/LongaBonga/Training-Classification-models-repo.git --branch=master

pip install -r requirements.txt --user

echo '~~~~~~~~~~~~SETTING UP SHARED FOLDERS~~~~~~~~~~'

ln -s /media/cluster_fs/user/dobryae/experiments/MobileNet/ ./experiments
ls -lh experiments/
ls -lh

DATA_FOLDER=/media/cluster_fs/datasets/classification/CIFAR100
export MODELS_ROOT=./experiments

WD="experiments/${EXPERIMENT_ID}"
mkdir $WD/artifacts

echo '~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~'
GPUS_NUM=`nvidia-smi -L | wc -l`
CMD="python -m torch.distributed.launch --nproc_per_node=$GPUS_NUM scripts/main.py --data_path ${DATA_FOLDER} \
              --model=mobilenet_v1 --output_dir=$WD/artifacts --fp16 --device=cuda --optimizer=sgd"
echo "$CMD"
$CMD