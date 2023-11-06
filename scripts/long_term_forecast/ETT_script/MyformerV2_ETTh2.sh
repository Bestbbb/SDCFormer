model_name=MyformerV2
seq_len=96
model_id_name=traffic
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecastingExperiment" ]; then
    mkdir .logs/LongForecastingExperiment
fi



for pred_len in 96 192 336 720
do
    python -u run.py \
      --task_name long_term_forecast \
      --batch_size 4 \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --itr 1 \
      --n_heads 4 \
      >logs/LongForecastingExperiment/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done