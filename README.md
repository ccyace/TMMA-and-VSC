cifar10 on ddim:
 训练 python scripts/two_stage_quantized_sampling.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --two_stage --quant_mode qdiff --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --cali_st 1 --cali_batch_size 32 --cali_n 4000 --cali_iters 20000 --cali_iters_a 5000 --cali_data_path_group0  --cali_data_path_group1  --max_avg_json "C:\Users\ASUS\Desktop\q-diffusion-master\activation_max_avg_stats.json" -l 

 采样 python scripts/split_sampling_from_ckpt.py --config configs/cifar10.yml --cali_ckpt  --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --skip_type quad --eta 0 --max_images 50000 --seed 1234

 
