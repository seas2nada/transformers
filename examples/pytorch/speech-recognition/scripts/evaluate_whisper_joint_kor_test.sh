torchrun \
	--nproc_per_node $GPUS run_speech_recognition_seq2seq.py \
	--model_name_or_path="whisper-large-joint-KlecSpeech-char-max10000" \
	--dataset_name=$data \
    --char_configs_path="./char-configs/korean" \
	--output_dir="whisper-large-joint-KlecSpeech-char-max10000" \
	--dataset_config_name=$data_config \
	--eval_split_name=$eval_subset \
	--per_device_eval_batch_size="48" \
	--generation_max_length="225" \
	--preprocessing_num_workers="48" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--audio_column_name=$audio_column_name \
	--text_column_name=$text_column_name \
    --language=$language \
    --return_ctc_logit="True" \
    --ctc_training \
	--group_by_length \
	--predict_with_generate \
	--fp16 \
	--do_eval \
	--use_auth_token \
    $extra_arguments
