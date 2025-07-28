CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset mrpc \
    --aggregation eclora \
    --model_name roberta-large \
    --seed 0 \
    --logging \
    --output_name default"