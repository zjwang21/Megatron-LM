LLAMA3_PATH=/home/nfs02/model/llama-3-8b
NUM_EXPERTS=4
TARGET_TP_SIZE=2
TARGET_EP_SIZE=2
TARGET_PP_SIZE=1
HF_FORMAT_DIR=/home/nfs04/wangzj/models/llama3-8b-mixtral-${NUM_EXPERTS}exp

echo "Upcycling llama3 to mixtral with ${NUM_EXPERTS} experts......"
python upcycling.py \
--model_path $LLAMA3_PATH \
--output_path $HF_FORMAT_DIR \
--num_experts $NUM_EXPERTS

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_FORMAT_DIR=/home/nfs04/wangzj/models/llama3-8b-mixtral-mcore-exp${NUM_EXPERTS}-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}

echo "Convert mixtral mdoel to megatron mcore model with tp=${TARGET_TP_SIZE} ep=${TARGET_EP_SIZE} pp=${TARGET_PP_SIZE}......"
cd ../
python tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_mixtral_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${LLAMA3_PATH}