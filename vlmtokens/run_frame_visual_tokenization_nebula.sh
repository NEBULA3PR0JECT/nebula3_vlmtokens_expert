### device config ###
N_GPU=1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"

### config ### 
DATASET=$1
SPLIT=$2
# DATASET="msrvtt"
# DATASET="youcook2"
# DATASET="vatex"
# DATASET="msvd"
# DATASET="vlep"
OUTPUT_ROOT_DIR=$3
SHARED_DATASETS="shared_datasets"

echo "running pipeline on dataset: $DATASET, ${SPLIT}"
echo "output root dir: $OUTPUT_ROOT_DIR"
echo "shared_datasets dir: $SHARED_DATASETS"


OUTPUT_DIR="$OUTPUT_ROOT_DIR/${DATASET}_${SPLIT}" # path to unique directory that will store all intermidiate and final results
CONFIG="configs/pipeline_config_msrvtt_test_nebula_toc.yaml"
VISUAL_TOKENIZATION_ENCODER="blip" # "clip" # "blip" 

VISUAL_TOKENIZATION_OUTPUT_DIR="$OUTPUT_DIR/visual_tokenization_$VISUAL_TOKENIZATION_ENCODER"
FRAME_CAPTION_OUTPUT_DIR="$OUTPUT_DIR/frame_caption"

mkdir -p $OUTPUT_DIR
mkdir -p $VISUAL_TOKENIZATION_OUTPUT_DIR
mkdir -p $FRAME_CAPTION_OUTPUT_DIR

# run visual tokenization 
if test -f "$VISUAL_TOKENIZATION_OUTPUT_DIR/visual_tokens.json"; then
    echo "visual tokens exist"
else
    echo "run visual tokenization..."
    # python -m torch.distributed.run --nproc_per_node=$N_GPU run_visual_tokenization.py \
    python run_visual_tokenization.py \
    --config $CONFIG \
    --output_dir $VISUAL_TOKENIZATION_OUTPUT_DIR \
    --encoder_version $VISUAL_TOKENIZATION_ENCODER
fi
