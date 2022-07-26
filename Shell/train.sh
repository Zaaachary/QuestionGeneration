# bsub -n 1 -q HPC.S1.GPU.X785.sha -o train.generation.0506-time.log -gpu num=1:mode=exclusive_process sh train_generation_zfli_tr.sh
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o run_logs/train/train.generation.0506-time.log -gpu num=1:mode=exclusive_process sh szcs/train_generation_zfli_tr.sh


nvidia-smi
cd ../Code

task_name='cpt-large'

init_model_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model"
output_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model"

PTM_name_or_path=$init_model_root/cpt-large/
# PTM_name_or_path=$init_model_root/cpt-base/

CUDA_VISIBLE_DEVICES=0 python run_train.py \
    --PTM_name_or_path $PTM_name_or_path \
    --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/QuestionGeneration/Data \
    --output_path $output_root/QuestionGeneration \
    --task_name $task_name \
    --max_src_len 128 \
    --max_tgt_len 30\
    --epoch 1 \
    --learning_rate 1e-5 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.0 \
    --warmup_proportion 0.025 \
    --train_batch_size_per_gpu 4 \
    --gradient_accumulation_step 2 \
    --dev_batch_size_per_gpu 32 \
    --seed 220406 \
    --gpus 0  \

