# bsub -n 1 -q HPC.S1.GPU.X785.sha -o train.generation.0506-time.log -gpu num=1:mode=exclusive_process sh train_generation_zfli_tr.sh
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o run_logs/train/train.generation.0506-time.log -gpu num=1:mode=exclusive_process sh szcs/train_generation_zfli_tr.sh


nvidia-smi

