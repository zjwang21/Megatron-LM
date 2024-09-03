cd ../
MEGATRON_PATCH_PATH=/home/nfs02/wangzj/public_code/Pai-Megatron-Patch
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240126
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
# 分别为训练集、验证集生成mmap格式预训练数据集。
python toolkits/pretrain_data_preprocessing/preprocess_data.py --jsonl-keys text \
    --input ./llamamoe/data \
    --patch-tokenizer-type LLamaTokenizer \
    --load /home/nfs04/wangzj/models/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
    --append-eod \
    --output-prefix ./llamamoe/data/mistral-tr1b \
    --dataset-impl mmap \
    --workers 16 \
    --seq-length 2048