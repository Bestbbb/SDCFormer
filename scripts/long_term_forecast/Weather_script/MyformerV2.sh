model_name=MyformerV2
seq_len=96
model_id_name=weather
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/test" ]; then
    mkdir ./logs/test
fi



for pred_len in 96 192 336 720
do
    python -u run.py \
      --task_name long_term_forecast \
      --batch_size 64 \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
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
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --n_heads 4 \
      >logs/LongForecastingExperiment/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done