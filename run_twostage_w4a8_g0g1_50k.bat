@echo off
title twostage w4a8 g0g1 cali+sample 50k
cd /d C:\Users\wsh\Desktop\q-diffusion-master
echo ========================================
echo Two-stage w4a8: g0/g1 BRECQ + correct DDIM split
echo g0: get_calibrations\out\dynamic_cali\cifar_g0_slice.pt
echo g1: get_calibrations\out\dynamic_cali\cifar_g1_slice.pt
echo ========================================
D:\Users\wsh\Anaconda3\envs\qdiff\python.exe scripts\two_stage_quantized_sampling.py --config configs/cifar10.yml --use_pretrained --seed 1234 --sample_type generalized --skip_type quad --timesteps 100 --eta 0 --ptq --two_stage --weight_bit 4 --act_bit 8 --quant_mode qdiff --quant_act --a_sym --split --cali_st 1 --cali_n 256 --cali_batch_size 32 --cali_iters 20000 --cali_iters_a 5000 --cali_data_path_group0 get_calibrations/out/dynamic_cali/cifar_g0_slice.pt --cali_data_path_group1 get_calibrations/out/dynamic_cali/cifar_g1_slice.pt --max_images 50000 -l twostage_w4a8_g0g1_50k
echo.
echo ExitCode=%ERRORLEVEL%
pause
