export WORK_DIR=/home/nfs02/wangzj/public_code
cd ${WORK_DIR}/Pai-Megatron-Patch/llamamoe

bash megatron.sh \
/home/nfs04/wangzj/models/mixtral-mcore-exp4-TP4PP1EP2 \
/home/nfs04/wangzj/models/mixtral-4exp/tokenizer.model \
$WORK_DIR/Pai-Megatron-Patch/llamamoe/data/mistral-tr1b_text_document \
1