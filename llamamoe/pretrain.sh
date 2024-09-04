export WORK_DIR=/home/nfs04/wangzj/code
cd ${WORK_DIR}/Megatron-LM/llamamoe
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash megatron.sh \
$WORK_DIR/Megatron-LM \
/home/nfs04/wangzj/models/llama3-8b-mixtral-mcore-exp4-TP2PP1EP2 \
/home/nfs02/model/llama-3-8b \
$WORK_DIR/Megatron-LM/llamamoe/data/stage2/entr5w \
2