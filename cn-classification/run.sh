export TASK_NAME=THUCNews

python main.py \
  --model_name_or_path bert-base-chinese \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./saved_outputs/$TASK_NAME/
