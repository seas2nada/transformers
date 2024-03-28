# --model_name_or_path="$PWD/whisper-large-500hFT" \

torchrun \
	--nproc_per_node $GPUS run_speech_recognition_seq2seq_peft.py \
	--model_name_or_path="whisper-large-librispeech-peft-max10000-lr1e-4-batch64" \
	--dataset_name=$data \
	--dataset_config_name=$data_config \
	--train_split_name=$train_subset \
	--eval_split_name=$eval_subset \
	--output_dir="whisper-large-librispeech-peft-max10000-lr1e-4-batch64" \
	--per_device_eval_batch_size="16" \
	--generation_max_length="225" \
	--preprocessing_num_workers="8" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--audio_column_name=$audio_column_name \
	--text_column_name=$text_column_name \
	--freeze_feature_encoder="False" \
	--ddp_timeout="999999" \
	--group_by_length \
	--predict_with_generate \
	--fp16 \
	--do_eval \
	--use_auth_token
