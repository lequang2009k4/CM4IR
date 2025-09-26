#!/bin/bash

# Số lượng GPU trên node (ở đây là 2 GPU)
gpu_count=2

# ID của task mapper hiện tại (do Hadoop phân phối)
task_id=$((${HADOOP_TASK_ID:-0}))  # Nếu không có task_id, mặc định là 0

# Gán GPU cho mỗi task mapper
gpu_id=$((task_id % gpu_count))    # Chỉ định GPU cho task mapper dựa trên task_id
export CUDA_VISIBLE_DEVICES=$gpu_id  # Chỉ định GPU cho task này

# Thư mục chứa ảnh trên local
source_dir="/kaggle/working/data"  # Thư mục chứa ảnh trên local

# Thư mục đích để lưu ảnh đã copy
target_dir="/kaggle/working/CM4IR/exp/datasets/lsun_cat/cat"  


# Đọc từng đường dẫn ảnh từ file input_list.txt
while read img; do
  [ -z "$img" ] && continue  # Bỏ qua dòng trống

  # Tên file
  fname=$(basename "$img")  # Lấy tên file (ví dụ: cat001.png)

  # Đường dẫn ảnh trong thư mục source_dir (local)
  source_img="$source_dir/$fname"


  # Copy ảnh từ thư mục source_dir vào thư mục đích
  mv "$source_img" "$target_dir/$fname" || { echo "FAIL MOVE $img" >&2; continue; }

  # Tạo run_id dựa trên tên file
  run_id="run_${fname%.*}"  # Ví dụ: run_cat001

done

# Chạy CM4IR trên ảnh đã copy vào thư mục đích
python3 /kaggle/working/CM4IR/main.py \
  --config /kaggle/working/CM4IR/configs/lsun_cat_256.yml \
  --path_y lsun_cat \
  --deg sr_bicubic \
  --deg_scale 4 \
  --sigma_y 0.05 \
  -i "$run_id" \
  --iN 250 \
  --gamma 0.2 \
  --model_ckpt lsun_cat/cd_cat256_lpips.pt || { echo "FAIL RUN $fname" >&2; rm -rf "$workdir"; continue; }

# Đường dẫn kết quả đầu ra
outdir="/kaggle/working/CM4IR/exp/image_samples/$run_id"

# Upload kết quả lên HDFS
hdfs dfs -put -f "$outdir"/*.png /result/

echo "DONE $run_id -> $outdir"

