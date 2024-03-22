
python run_speech_recognition_seq2seq.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name=$data \
	--dataset_config_name=$data_config \
	--language=$language \
	--train_split_name=$train_subset \
	--eval_split_name=$eval_subset \
	--max_steps="5000" \
	--output_dir="./whisper-base-libri" \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="2" \
	--per_device_eval_batch_size="16" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name=$text_column_name \
	--freeze_feature_encoder="False" \
	--gradient_checkpointing \
	--group_by_length \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token
