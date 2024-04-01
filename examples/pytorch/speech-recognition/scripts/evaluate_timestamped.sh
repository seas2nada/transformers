torchrun \
	--nproc_per_node $GPUS run_speech_recognition_seq2seq_with_timestamps.py \
	--model_name_or_path="openai/whisper-large-v3" \
	--dataset_name=$data \
	--dataset_config_name=$data_config \
	--eval_split_name=$eval_subset \
	--output_dir="./whisper-large-v3" \
	--per_device_eval_batch_size="16" \
	--generation_max_length="225" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--audio_column_name=$audio_column_name \
	--text_column_name=$text_column_name \
	--freeze_feature_encoder="False" \
	--predict_timestamps="False" \
	--language=$language \
	--ddp_timeout="999999" \
	--group_by_length \
	--predict_with_generate \
	--fp16 \
	--do_eval \
	--use_auth_token \
    $extra_arguments
