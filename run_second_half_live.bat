@echo off
title second-half live PNG w4a8 stage2
cd /d C:\Users\wsh\Desktop\q-diffusion-master
set LOGDIR=C:\Users\wsh\Desktop\q-diffusion-master\twostage_w4a8_g0g1_50k\samples\2026-07-29-12-31-29
echo ========================================
echo Second-half only, save PNG each batch
echo ckpt : %LOGDIR%\ckpt_stage2_group1.pth
echo inter: %LOGDIR%\intermediate_noise_images.pt
echo out  : %LOGDIR%\images_live\
echo ========================================
if not exist "%LOGDIR%\images_live" mkdir "%LOGDIR%\images_live"
D:\Users\wsh\Anaconda3\envs\qdiff\python.exe scripts\split_sampling_from_ckpt.py --config configs/cifar10.yml --cali_ckpt "%LOGDIR%\ckpt_stage2_group1.pth" --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --second_half_only --intermediate_path "%LOGDIR%\intermediate_noise_images.pt" --image_folder "%LOGDIR%\images_live" --fresh_images --max_images 50000 --batch_size 64 --skip_type quad --eta 0 --seed 1234
echo.
echo ExitCode=%ERRORLEVEL%
pause
