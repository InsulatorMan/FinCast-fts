
#data
data_folder=./datasets/stock_v1/v1_no_volume
val_data_folder=./datasets/stock_v1/val_v1_nv
num_workers=4
freq_map_mode=1

#training, batch size maximum 512 for 4090, around 20gb vram. 256 is optimum for 1 card
batch_size=32
num_epochs=3

#others
wandb_project=Fincast_local

#model  back up options     --use_amp
checkpoint=./checkpoints/new_base_l2/trained_best_model_ffm_medium_E4T2_epoch5_full_ts_major6.pth
num_experts=4
gating_top_n=2

#remember to change this
model_save_dir=checkpoints/t1/Fincast_local_4090_t1

#logging
log_every_n_steps=100
logging_dir=logs/Fincast/t1

#loss field
loss_slope_weight=0.8
quantile_loss_coef=0.04
grad_max_norm=1.0
loss_delta=1.0

aux_loss_coef=0.01
learning_rate=0.00001
weight_decay=0.01

python -u training/Training_DDP.py \
    --data_folder $data_folder \
    --val_data_folder $val_data_folder \
    --freq_map_mode $freq_map_mode \
    --num_workers $num_workers \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --wandb_project $wandb_project \
    --distributed \
    --save_best \
    --load_from_compile \
    --use_quantile_loss \
    --loss_slope_weight $loss_slope_weight \
    --quantile_loss_coef $quantile_loss_coef \
    --use_moe_aux_loss \
    --torch_compile \
    --series_norm \
    --use_wandb \
    --mask_ratio 0.0 \
    --log_every_n_steps $log_every_n_steps \
    --logging_dir $logging_dir \
    --checkpoint $checkpoint \
    --num_experts $num_experts \
    --gating_top_n $gating_top_n \
    --aux_loss_coef $aux_loss_coef \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --grad_max_norm $grad_max_norm \
    --loss_delta $loss_delta \
    --model_save_dir $model_save_dir