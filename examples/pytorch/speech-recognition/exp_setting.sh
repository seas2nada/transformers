# Activate venv
source $PWD/../../../tools/venv/bin/activate

# Environment setting
export GPUS=2

# Experiment setting
export data=KlecSpeech    # librispeech, commonvoice
num_proc=16
extra_arguments=()

if [[ "$data" == "librispeech" ]]; then
    export data=librispeech_asr
    export data_config="all"  # clean, other 
    export train_subset="train.clean.100+train.clean.360+train.other.500" # clean: train.100, train.360 / other: train.500
    export eval_subset="test.clean"   # test, validation
    export audio_column_name="audio"
    export text_column_name="text"
    export language="english"
elif [[ "$data" == "cv" ]]; then
    export data=mozilla-foundation/common_voice_16_0
    export data_config="ko"  # Language ID
    export train_subset="train" # train
    export eval_subset="validation"   # test, validation
    export audio_column_name="audio"
    export text_column_name="sentence"
    export language="korean"
elif [[ "$data" == "ksponspeech" ]]; then
    export train_subset="train" # train
    export eval_subset="validation"   # test, validation
    export text_column_name="sentence"
    export audio_column_name="audio"
    export language="korean"
    export num_proc=128
    export data_dir="/home/ubuntu/Workspace/DB/LibriSpeech/KsponSpeech"
    export from_local_disk="/home/ubuntu/Workspace/DB/LibriSpeech/KsponSpeech/arrows"
elif [[ "$data" == "zeroth" ]]; then
    export data=Bingsu/zeroth-korean
    export train_subset="train" # train
    export eval_subset="test"   # test
    export text_column_name="text"
    export audio_column_name="audio"
    export language="korean"
elif [[ "$data" == "KlecSpeech" ]]; then
    export data=data/KlecSpeech
    export train_subset="train" # train
    export eval_subset="validation"   # test
    export text_column_name="text"
    export audio_column_name="audio"
    export language="korean"
    export num_proc=64
    extra_arguments+=" --from_json"
    extra_arguments+=" --preprocessing_cache_file_dir ~/.cache/huggingface/datasets/KlecSpeech_cache"
fi
