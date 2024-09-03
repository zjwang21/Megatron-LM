TOKENIZER_MODEL=/home/nfs04/wangzj/models/mixtral-2exp/tokenizer.model
MEGATRON_PATH="/home/nfs02/wangzj/public_code/Pai-Megatron-Patch/Megatron-LM"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE="4"
TARGET_EP_SIZE="2"
TARGET_PP_SIZE="1"

HF_FORMAT_DIR=/home/nfs04/wangzj/models/mixtral-4exp
MEGATRON_FORMAT_DIR=/home/nfs04/wangzj/models/mixtral-mcore-exp4-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}

cd ../Megatron-LM
python tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_mixtral_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL}