export WORK_DIR=/home/nfs04/wangzj/code/Megatron-LM

python $WORK_DIR/tools/hf_pretrain_dataset_patch/preprocess_pretrain_data_hf.py \
       --data_dir $WORK_DIR/llamamoe/data/stage2 \
       --save_dir $WORK_DIR/llamamoe/data/stage2/entr5w \
       --tokenizer_name_or_path /home/nfs02/model/llama-3-8b \
       --sequence_length 1024