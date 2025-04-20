is_finetune=1
is_transfer=1

dset_finetune='SWAT'
context_points=1056
target_points=96
batch_size=64
is_half=1

e_layers=6
d_layers=6

patch_len=48
stride=48

revin=1

model_type='LightGTS_base_full_shot'
pretrained_model='/home/LightGTS/checkpoints/LightGTS_8M.pth'

n_epochs_finetune=20
n_epochs_freeze=20


# random_seed=2021


for dset_finetune in 'etth1' 'etth2' 
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Forecasting" ]; then
        mkdir logs/Forecasting
    fi
    if [ ! -d "logs/Forecasting/$model_type" ]; then
        mkdir logs/Forecasting/$model_type
    fi
    if [ ! -d "logs/LongForecasting/$model_type/$dset_finetune" ]; then
        mkdir logs/Forecasting/$model_type/$dset_finetune
    fi

    for target_points in 96 192 336 720
    do
        python -u finetune.py \
        --is_finetune $is_finetune \
        --is_transfer $is_transfer \
        --dset_finetune $dset_finetune \
        --is_half $is_half \
        --context_points 528 \
        --target_points $target_points \
        --batch_size $batch_size \
        --patch_len 48\
        --stride 48\
        --revin 1 \
        --e_layers $e_layers\
        --d_layers $d_layers\
        --n_heads 8 \
        --d_model 256 \
        --d_ff 512\
        --dropout 0.2\
        --head_drop 0.2 \
        --n_epochs_finetune $n_epochs_finetune\
        --n_epochs_freeze $n_epochs_freeze\
        --lr 1e-4 \
        --finetuned_model_id 1\
        --pretrained_model $pretrained_model\
        --model_type $model_type\  >logs/Forecasting/$model_type/$dset_finetune/'percentage'$is_half'_finetune'$is_finetune'_context'$context_points'_target'$target_points.log 
    done
done

for dset_finetune in 'ettm1' 'ettm2' 
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Forecasting" ]; then
        mkdir logs/Forecasting
    fi
    if [ ! -d "logs/Forecasting/$model_type" ]; then
        mkdir logs/Forecasting/$model_type
    fi
    if [ ! -d "logs/LongForecasting/$model_type/$dset_finetune" ]; then
        mkdir logs/Forecasting/$model_type/$dset_finetune
    fi

    for target_points in 96 192 336 720
    do
        python -u finetune.py \
        --is_finetune $is_finetune \
        --is_transfer $is_transfer \
        --dset_finetune $dset_finetune \
        --is_half $is_half \
        --context_points 2112 \
        --target_points $target_points \
        --batch_size $batch_size \
        --patch_len 192\
        --stride 192\
        --revin 1 \
        --e_layers $e_layers\
        --d_layers $d_layers\
        --n_heads 8 \
        --d_model 256 \
        --d_ff 512\
        --dropout 0.2\
        --head_drop 0.2 \
        --n_epochs_finetune $n_epochs_finetune\
        --n_epochs_freeze $n_epochs_freeze\
        --lr 1e-4 \
        --finetuned_model_id 1\
        --pretrained_model $pretrained_model\
        --model_type $model_type\  >logs/Forecasting/$model_type/$dset_finetune/'percentage'$is_half'_finetune'$is_finetune'_context'$context_points'_target'$target_points.log 
    done
done

for dset_finetune in  'weather' 'solar'
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Forecasting" ]; then
        mkdir logs/Forecasting
    fi
    if [ ! -d "logs/Forecasting/$model_type" ]; then
        mkdir logs/Forecasting/$model_type
    fi
    if [ ! -d "logs/LongForecasting/$model_type/$dset_finetune" ]; then
        mkdir logs/Forecasting/$model_type/$dset_finetune
    fi

    for target_points in 96 192 336 720
    do
        python -u finetune.py \
        --is_finetune $is_finetune \
        --is_transfer $is_transfer \
        --dset_finetune $dset_finetune \
        --is_half $is_half \
        --context_points 2880 \
        --target_points $target_points \
        --batch_size $batch_size \
        --patch_len 48\
        --stride 48\
        --revin 1 \
        --e_layers $e_layers\
        --d_layers $d_layers\
        --n_heads 8 \
        --d_model 256 \
        --d_ff 512\
        --dropout 0.2\
        --head_drop 0.2 \
        --n_epochs_finetune $n_epochs_finetune\
        --n_epochs_freeze $n_epochs_freeze\
        --lr 1e-4 \
        --finetuned_model_id 1\
        --pretrained_model $pretrained_model\
        --model_type $model_type\  >logs/Forecasting/$model_type/$dset_finetune/'percentage'$is_half'_finetune'$is_finetune'_context'$context_points'_target'$target_points.log 
    done
done

for dset_finetune in  'electricity' 'traffic'
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Forecasting" ]; then
        mkdir logs/Forecasting
    fi
    if [ ! -d "logs/Forecasting/$model_type" ]; then
        mkdir logs/Forecasting/$model_type
    fi
    if [ ! -d "logs/LongForecasting/$model_type/$dset_finetune" ]; then
        mkdir logs/Forecasting/$model_type/$dset_finetune
    fi

    for target_points in 96 192 336 720
    do
        python -u finetune.py \
        --is_finetune $is_finetune \
        --is_transfer $is_transfer \
        --dset_finetune $dset_finetune \
        --is_half $is_half \
        --context_points 528 \
        --target_points $target_points \
        --batch_size 32 \
        --patch_len 48\
        --stride 48\
        --revin 1 \
        --e_layers $e_layers\
        --d_layers $d_layers\
        --n_heads 8 \
        --d_model 256 \
        --d_ff 512\
        --dropout 0.2\
        --head_drop 0.2 \
        --n_epochs_finetune $n_epochs_finetune\
        --n_epochs_freeze $n_epochs_freeze\
        --lr 1e-4 \
        --finetuned_model_id 1\
        --pretrained_model $pretrained_model\
        --model_type $model_type\  >logs/Forecasting/$model_type/$dset_finetune/'percentage'$is_half'_finetune'$is_finetune'_context'$context_points'_target'$target_points.log 
    done
done