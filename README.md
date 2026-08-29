cifar10 on ddim:
 训练 python scripts/two_stage_quantized_sampling.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --two_stage --quant_mode qdiff --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --cali_st 1 --cali_batch_size 32 --cali_n 4000 --cali_iters 20000 --cali_iters_a 5000 --cali_data_path_group0  --cali_data_path_group1  --max_avg_json "C:\Users\ASUS\Desktop\q-diffusion-master\activation_max_avg_stats.json" -l 

 采样 python scripts/split_sampling_from_ckpt.py --config configs/cifar10.yml --cali_ckpt  --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --skip_type quad --eta 0 --max_images 50000 --seed 1234

bedroom64*64 on DDIM
训练:C:\Users\ASUS\miniconda3\envs\qdiff\python.exe scripts/two_stage_quantized_sampling.py --config configs/lsun_bedroom256.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --two_stage --quant_mode qdiff --weight_bit 8 --act_bit 8 --quant_act --a_sym --split --cali_st 1 --cali_batch_size 1 --cali_n 8 --cali_iters 20000 --cali_iters_a 5000 --cali_data_path_group0  --cali_data_path_group1 " --max_avg_json  --max_images  -l 

采样python scripts/split_sampling_from_ckpt.py --config configs/cifar10.yml --cali_ckpt  --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --skip_type quad --eta 0 --max_images 50000 --seed 1234

 LDM_church权重通过网盘分享的文件：LDM_church
链接: https://pan.baidu.com/s/1jLthCHD01XNBgsxLACGbjg?pwd=nnm7 提取码: nnm7 
--来自百度网盘超级会员v1的分享

C:\Users\ASUS\miniconda3\envs\qdiff\python.exe scripts\sample_diffusion_ldm.py -r models/ldm/lsun_churches256/model.ckpt -n 2 --batch_size 1 -c 400 -e 0.0 --seed 40 --ptq --two_stage --quant_mode qdiff --weight_bit 8 --quant_act --act_bit 8 --a_sym --cali_st 20 --cali_n 64 --cali_batch_size 1 --cali_iters 2000 --cali_iters_a 500 --skip_type quad --cali_data_path get_calibrations/out/dynamic_cali/lsun_church_ldm_ddim400_sample256_allst.pt --max_avg_json church_ldm_layerwise_activation_statistics.json -l 
