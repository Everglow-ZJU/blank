export MODEL_NAME="audioldm-m-full"
# export INSTANCE_DIR="path/to/concept/audios" #instance data_dir
# export OUTPUT_DIR="path/to/output/dir"

idx=$1
prompt_idx=$((idx % 21))
class_idx=$((idx / 21))

unique_token="sks"
subject_names=(
    "agni_parthene" "amen" "bagpipe" "bouzouki" "cajon"
    "didgeridoo" "eminem" "epirus" "gtr" "hiphop"
    "jaws" "jingle" "kaoru" "kashaka" "morricone"
    "ocarina" "ode" "oud" "rabab" "reggae"
    "rumaba" "sarabande" "shehnai" "shofar" "sitar"
    "toere" "tsifteteli" "tuvan" "vader" "zoumbas"
)

class_tokens=(
    "chant" "drum beat" "wind instrument" "string instrument" "percussion"
    "wind instrument" "rapper" "singers" "guitar" "drum beat"
    "bass" "guitar" "saxophone" "percussion" "classical music"
    "wind instrument" "classical music" "string instrument" "string instrument" "drum beat"
    "drum beat" "guitar" "wind instrument" "wind instrument" "string instrument"
    "percussion" "drum beat" "singer" "classical music" "violin"
)
class_token=${class_tokens[$class_idx]}
selected_subject=${subject_names[$class_idx]}

text_editability_templates=(
    "a jazz song with a ${unique_token} ${class_token}"
    "a rock song with a ${unique_token} ${class_token}"
    "a disco song with a ${unique_token} ${class_token}"
    "a techno song with a ${unique_token} ${class_token}"
    "a heavy metal song with a ${unique_token} ${class_token}"
    "a recording of a ${unique_token} ${class_token} in a small room"
    "a recording of a ${unique_token} ${class_token} in a cathedral"
    "a recording of a ${unique_token} ${class_token} under water"
    "a gramophone recording of a ${unique_token} ${class_token}"
    "a recording of a ${unique_token} ${class_token} song with a rock drum beat accompaniment"
    "a recording of a ${unique_token} ${class_token} song with a jazz drum beat accompaniment"
    "a recording of a ${unique_token} ${class_token} song with a hip hop drum beat accompaniment"
    "a recording of a ${unique_token} ${class_token} song with a man singing"
    "a recording of a ${unique_token} ${class_token} in the rain"
    "a recording of a ${unique_token} ${class_token} with footsteps"
    "a recording of a ${unique_token} ${class_token} in a big crowd"
    "a recording of a vibrant ${unique_token} ${class_token} solo with a classical guitar accompaniment"
    "a recording of a ${unique_token} ${class_token} with a synth baseline accompaniment"
    "a recording of a ${unique_token} ${class_token} with a saxophone accompaniment"
    "a recording of a ${unique_token} ${class_token} playing a happy tune"
    "a recording of a ${unique_token} ${class_token} playing a sad tune"
    )
#多概念直接用就行
validation_prompt=${text_editability_templates[$prompt_idx]}
name="${selected_subject}-${prompt_idx}"
#prompt used for training
# instance_prompt="a recording of ${unique_token} ${class_token}"
# class_prompt="a recording of ${class_token}"

export OUTPUT_DIR="log_dreamsound/${name}"
export INSTANCE_DIR="../data/dreambooth/${selected_subject}"
export CLASS_DIR="data/class_data/${class_token}"

accelerate launch dreambooth_audioldm.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$INSTANCE_DIR \
    --class_data_dir="$CLASS_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --with_prior_preservation --prior_loss_weight=1.0 \ 
    --duration=10.0 \
    --instance_word=$unique_token \
    --object_class=$class_token \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=100 \
    --learning_rate=4.0e-06 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1500 \
    --validation_prompt="$validation_prompt" \
    --validation_epochs=1 \
    --output_dir=$OUTPUT_DIR \
    --num_vectors=1 \
    --seed="0" \
    --num_class_audio_files=100 \
    --save_as_full_pipeline \
    
