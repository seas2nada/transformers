. ../../../tools/activate_python.sh
# Environment setting
export GPUS=4

# Experiment setting
data=librispeech    # librispeech, commonvoice
# export train_subset="train.100+train.360" # clean: train.100, train.360 / other: train.500
# export eval_subset="test"   # test, validation
# LibriSpeech
if [[ "$data" == "librispeech" ]]; then
    export data=librispeech_asr
    export data_config="all"  # clean, other 
    export train_subset="train.clean.100+train.clean.360+train.other.500" # clean: train.100, train.360 / other: train.500
    export eval_subset="validation.clean"   # test, validation
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
fi
