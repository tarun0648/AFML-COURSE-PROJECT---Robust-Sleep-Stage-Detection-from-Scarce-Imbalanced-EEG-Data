exp="SSL_SimCLR_1DCNN"       # MUST match
train_mode="ft_5per"
device="cpu"
data_percentage="5"
sleep_model="cnn1d"
ssl_method="simclr"         

python3 main.py \
    --device $device \
    --experiment_description $exp \
    --run_description "Pretrain_Fold_0" \
    --fold_id 0 \
    --train_mode $train_mode \
    --data_percentage $data_percentage \
    --sleep_model $sleep_model \
    --ssl_method $ssl_method \
    --augmentation "noise_permute" \
    --dataset "sleep_edf"