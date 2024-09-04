cd ../
python tools/preprocess_data.py \
       --input /home/nfs02/wangzj/aliyun/tr_part1b_00000.jsonl \
       --output-prefix llamamoe/data/culturax-tr1b-mistral \
       --tokenizer-model /home/nfs02/model/llama-3-8b \
       --tokenizer-type HuggingFaceTokenizer \
       --append-eod \
       --workers 16