torchrun \
 	--nproc_per_node $GPUS run_speech_recognition_seq2seq_with_timestamps.py \
	--model_name_or_path="openai/whisper-large-v3" \
	--dataset_name=$data \
	--dataset_config_name=$data_config \
	--train_split_name=$train_subset \
	--eval_split_name=$eval_subset \
	--num_train_epochs="2" \
    --load_from_json="True" \
	--output_dir="./whisper-large-KsponSpeech-TS-withdedup-2epoch-batch256" \
	--eval_metric="wer" \
	--metric_for_best_model="wer" \
	--load_best_model_at_end="True" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--ddp_timeout "9999999" \
	--freeze_feature_encoder "False" \
	--audio_column_name=$audio_column_name \
	--text_column_name=$text_column_name \
	--preprocessing_num_workers=$num_proc \
	--language="korean" \
	--preprocessing_only="False" \
	--gradient_checkpointing \
	--group_by_length \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token \
	$extra_arguments
