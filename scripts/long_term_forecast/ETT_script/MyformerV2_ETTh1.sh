model_name=MyformerV2
seq_len=96
model_id_name=ETTh1
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
      --batch_size 128 \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --n_heads 2 \
      >logs/LongForecastingExperiment/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done