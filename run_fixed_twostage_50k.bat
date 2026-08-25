@echo off
title FIXED two-stage sample 50k
cd /d C:\Users\wsh\Desktop\q-diffusion-master
set LOGDIR=C:\Users\wsh\Desktop\q-diffusion-master\twostage_w4a8_g0g1_50k\samples\2026-07-29-12-31-29
set OUT=%LOGDIR%\images_fixed_50k
set INTER=%LOGDIR%\intermediate_noise_images_fixed_50k.pt
if not exist "%OUT%" mkdir "%OUT%"
echo ========================================
echo FIXED two-stage sampling to 50000 images
echo [1] stage1 first-half limited 50 steps
echo [2] stage2 second-half, PNG each batch
echo out: %OUT%
echo ========================================
echo [1/2] first half ...
D:\Users\wsh\Anaconda3\envs\qdiff\python.exe scripts\split_sampling_from_ckpt.py --config configs/cifar10.yml --cali_ckpt "%LOGDIR%\ckpt_stage1_group0.pth" --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --first_half_only --overwrite_intermediate --intermediate_path "%INTER%" --max_images 50000 --batch_size 64 --skip_type quad --eta 0 --seed 1234
if errorlevel 1 goto :fail
echo [2/2] second half live PNG ...
D:\Users\wsh\Anaconda3\envs\qdiff\python.exe scripts\split_sampling_from_ckpt.py --config configs/cifar10.yml --cali_ckpt "%LOGDIR%\ckpt_stage2_group1.pth" --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --second_half_only --intermediate_path "%INTER%" --image_folder "%OUT%" --fresh_images --max_images 50000 --batch_size 64 --skip_type quad --eta 0 --seed 1234
echo.
echo ExitCode=%ERRORLEVEL%
echo PNG: %OUT%
pause
exit /b %ERRORLEVEL%
:fail
echo FIRST HALF FAILED
pause
exit /b 1
