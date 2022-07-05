# bsub -n 1 -q HPC.S1.GPU.X785.sha -o infer.log -gpu num=1:mode=exclusive_process sh infer.sh

nvidia-smi
cd ../Code

PTM_name_or_path="../Model/init_model_cpt-large/"

CUDA_VISIBLE_DEVICES=0 python run_infer.py \
    --PTM_name_or_path $PTM_name_or_path \
    --input_path ../Data/dev.jsonl \
    --model_path ../Model/bz=1x4x2_ep=1_lr=1e-05_ae=1e-06_seed=220406/checkpoints/epoch=00-step=1814-val_loss=1.068272.ckpt \
    --output_path ../Model/bz=1x4x2_ep=1_lr=1e-05_ae=1e-06_seed=220406/predict.json\
    --device 0 \
    --max_src_len 128 \
    --eval_length 30

# beam_nums 一对 Passage + Answer 产出的问题数量
# device 0:0号GPU, -1:使用CPU