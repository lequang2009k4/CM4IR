## Experiments on LSUN Bedroom ##
#---- Noise level 0.025 ----
# Super-Resolution Bicubic x4
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg sr_bicubic --deg_scale 4 \
  --sigma_y 0.025 -i CM4IR_lsun_bedroom_sr_bicubic_sigma_y_0.025 --iN 400 --gamma 0.7 \
  --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt #--deltas "0,0.5,0.1,0"

# Gaussian Deblurring
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg deblur_gauss \
  --sigma_y 0.025 -i CM4IR_lsun_bedroom_deblur_gauss_sigma_y_0.025 --iN 75 --gamma 0.01 \
  --zeta 3 --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt #--deltas "0.1,0,0,0.05"

# Inpainting (Random, 80%)
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg inpainting \
  --sigma_y 0.025 -i CM4IR_lsun_bedroom_inpainting_random_80_sigma_y_0.025 --iN 150 --gamma 0.2 \
  --inpainting_mask_path random_80_mask.npy --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt # --deltas "0.1,0.1,0.8,0.8" --deltas_injection_type 1

# Inpainting (Letters)
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg inpainting \
  --sigma_y 0.025 -i CM4IR_lsun_bedroom_inpainting_letters_sigma_y_0.025 --iN 150 --gamma 0.2 \
  --inpainting_mask_path lorem_ipsum_mask.npy --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt # --deltas "0.2,0.3,0.8,0.8" --deltas_injection_type 1

#---- Noise level 0.05 ----
# Super-Resolution Bicubic x4
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg sr_bicubic --deg_scale 4 \
  --sigma_y 0.05 -i CM4IR_lsun_bedroom_sr_bicubic_sigma_y_0.05 --iN 250 --gamma 0.2 \
  --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt #--deltas "0,0.3,0.05,0.1"

# Gaussian Deblurring
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg deblur_gauss \
  --sigma_y 0.05 -i CM4IR_lsun_bedroom_deblur_gauss_sigma_y_0.05 --iN 150 --gamma 0.05 \
  --zeta 1 --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt #--deltas "0.1,0,0,0.05"

# Inpainting (Random, 80%)
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg inpainting \
  --sigma_y 0.05 -i CM4IR_lsun_bedroom_inpainting_random_80_sigma_y_0.05 --iN 150 --gamma 0.2 \
  --inpainting_mask_path random_80_mask.npy --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt # --deltas "0.2,0.3,0.8,0.8" --deltas_injection_type 1

#---- Noiseless ----
# Inpainting (Random, 80%)
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg inpainting \
  --sigma_y 0.0 -i CM4IR_lsun_bedroom_inpainting_random_80_sigma_y_0.0 --iN 150 --gamma 0.2 \
  --inpainting_mask_path random_80_mask.npy --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt # --deltas "0.2,0.1,1.0,1.0" --deltas_injection_type 1

## Experiments on LSUN Cat ##
#---- Noise level 0.025 ----
# Super-Resolution Bicubic x4
python main.py --config lsun_cat_256.yml --path_y lsun_cat --deg sr_bicubic --deg_scale 4 \
  --sigma_y 0.025 -i CM4IR_lsun_cat_sr_bicubic_sigma_y_0.025 --iN 400 --gamma 0.7 \
  --model_ckpt lsun_cat/cd_cat256_lpips.pt #--deltas "0,0.3,0,0"

# Gaussian Deblurring
python main.py --config lsun_cat_256.yml --path_y lsun_cat --deg deblur_gauss \
  --sigma_y 0.025 -i CM4IR_lsun_cat_deblur_gauss_sigma_y_0.025 --iN 125 --gamma 0.05 \
  --zeta 4 --model_ckpt lsun_cat/cd_cat256_lpips.pt

# Inpainting (Random, 80%)
python main.py --config lsun_cat_256.yml --path_y lsun_cat --deg inpainting \
  --sigma_y 0.025 -i CM4IR_lsun_cat_inpainting_random_80_sigma_y_0.025 --iN 150 --gamma 0.2 \
  --inpainting_mask_path random_80_mask.npy --model_ckpt lsun_cat/cd_cat256_lpips.pt # --deltas "0,0,1.0,1.0" --deltas_injection_type 1

#---- Noise level 0.05 ----
# Super-Resolution Bicubic x4
python main.py --config lsun_cat_256.yml --path_y lsun_cat --deg sr_bicubic --deg_scale 4 \
  --sigma_y 0.05 -i CM4IR_lsun_cat_sr_bicubic_sigma_y_0.05 --iN 250 --gamma 0.2 \
  --model_ckpt lsun_cat/cd_cat256_lpips.pt #--deltas "0.1,0.1,0,0"

# Gaussian Deblurring
python main.py --config lsun_cat_256.yml --path_y lsun_cat --deg deblur_gauss \
  --sigma_y 0.05 -i CM4IR_lsun_cat_deblur_gauss_sigma_y_0.05 --iN 250 --gamma 0.2 \
  --zeta 1 --model_ckpt lsun_cat/cd_cat256_lpips.pt

# Inpainting (Random, 80%)
python main.py --config lsun_cat_256.yml --path_y lsun_cat --deg inpainting \
  --sigma_y 0.05 -i CM4IR_lsun_cat_inpainting_random_80_sigma_y_0.05 --iN 150 --gamma 0.2 \
  --inpainting_mask_path random_80_mask.npy --model_ckpt lsun_cat/cd_cat256_lpips.pt # --deltas "0,0,1.0,1.0" --deltas_injection_type 1

## Experiments on ImageNet64 ##
#---- Noise level 0.01 ----
# Super-Resolution Bicubic x2
python main.py --config imagenet_64_cc.yml --path_y imagenet --deg sr_bicubic --deg_scale 2 \
  --sigma_y 0.01 -i CM4IR_imagenet_sr_bicubic_sigma_y_0.01 --iN 50 --gamma 0.01 \
  --model_ckpt imagenet/cd_imagenet64_lpips.pt

#---- Noise level 0.025 ----
# Super-Resolution Bicubic x2
python main.py --config imagenet_64_cc.yml --path_y imagenet --deg sr_bicubic --deg_scale 2 \
  --sigma_y 0.025 -i CM4IR_imagenet_sr_bicubic_sigma_y_0.025 --iN 75 --gamma 0.02 \
  --model_ckpt imagenet/cd_imagenet64_lpips.pt

#---- Noise level 0.05 ----
# Super-Resolution Bicubic x2
python main.py --config imagenet_64_cc.yml --path_y imagenet --deg sr_bicubic --deg_scale 2 \
  --sigma_y 0.05 -i CM4IR_imagenet_sr_bicubic_sigma_y_0.05 --iN 100 --gamma 0.03 \
  --model_ckpt imagenet/cd_imagenet64_lpips.pt
