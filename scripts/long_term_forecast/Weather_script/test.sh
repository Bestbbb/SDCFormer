


for pred_len in 24 48 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --batch_size 32 \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model MyformerV2 \
    --data custom \
    --features M \
    --seq_len 96 \
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
    >logs/LongForecasting/MyformerV2TEST_Weather_96'_'$pred_len.log

done
