export MODEL_NAME="runwayml/stable-diffusion-v1-5"

# 清空或创建日志文件（可选）
> custom_output.log

# 循环迭代 idx 从 0 到 749
for idx in {0..749}; do
    nohup python eval_ckpt.py \
     --idx=$idx \
     --pretrained_model_name_or_path=$MODEL_NAME \
     --num_validation_images=8 \
     --seed="0" \
     --epoch=4 \
     --output_root_path="./log_fft" \
     >> custom_output.log 2>&1 &  # 使用 >> 追加写入日志文件
done
