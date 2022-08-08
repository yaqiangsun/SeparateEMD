
python train.py  --max_epoch 60 --model_class SeparateEMD  --backbone_class Res12 --dataset MiniImageNet --way 5 \
 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --num_eval_episodes 500 --balance 0.1 --temperature 12.5 \
 --temperature2 32 --lr 0.0005 --lr_mul 10 --lr_scheduler step --step_size 10 --gamma 0.5 --gpu 0,1,2,3 \
 --init_weights ./saves/initialization/miniimagenet/Res12-pre-feat.pth \
 --eval_interval 1 --use_euclidean --solver opencv --no_avg_pool --not_random_val_task --save_flag fixSeedOriginSamplingAugDoubleMean3_2 \
 --augment \
 --deepemd sampling \
 --multi_gpu \
 --num_patch 9
#  --weight_decay 0.0005 \
# --fix_BN --finetune --resume_weights checkpoints/MiniImageNet-SeparateEMD-Res12-05w01s15q-Pre-DIS/20_0.5_lr0.0002mul10_step_T112.5T232.0_b0.1_bsz080-NoAugNonefirst/max_acc.pth \
#  --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth \