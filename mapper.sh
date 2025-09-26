#!/bin/bash

# Số lượng GPU trên node (ở đây là 2 GPU)
gpu_count=2

# ID của task mapper hiện tại (do Hadoop phân phối)
task_id=$((${HADOOP_TASK_ID:-0}))  # Nếu không có task_id, mặc định là 0

# Gán GPU cho mỗi task mapper
gpu_id=$((task_id % gpu_count))    # Chỉ định GPU cho task mapper dựa trên task_id
export CUDA_VISIBLE_DEVICES=$gpu_id  # Chỉ định GPU cho task này

# Đọc từng đường dẫn ảnh từ file input_list.txt
while read img; do
  [ -z "$img" ] && continue  # Bỏ qua dòng trống

  # Tên file và thư mục tạm
  fname=$(basename "$img")
  stem="${fname%.*}"
  workdir="/tmp/cm4ir_${stem}_$$"
  mkdir -p "$workdir"

  # Lấy ảnh từ thư mục giả lập HDFS về local (Giả sử ảnh đã được tải lên HDFS)
  hdfs dfs -get -f "$img" "$workdir/$fname" || { echo "FAIL GET $img" >&2; rm -rf "$workdir"; continue; }

  run_id="run_${stem}"

  # Chạy CM4IR trên ảnh
  python3 /kaggle/working/CM4IR/main.py \
    --config /kaggle/working/CM4IR/configs/lsun_cat_256.yml \
    --path_y "$workdir/$fname" \
    --deg sr_bicubic \
    --deg_scale 4 \
    --sigma_y 0.05 \
    -i "$run_id" \
    --iN 250 \
    --gamma 0.2 \
    --model_ckpt lsun_cat/cd_cat256_lpips.pt || { echo "FAIL RUN $fname" >&2; rm -rf "$workdir"; continue; }

  outdir="/kaggle/working/CM4IR/exp/image_samples/$run_id"

  # Upload kết quả lên HDFS
  hdfs dfs -put -f "$outdir"/*.png /result/

  echo "DONE $fname -> $outdir"
  rm -rf "$workdir"
done
