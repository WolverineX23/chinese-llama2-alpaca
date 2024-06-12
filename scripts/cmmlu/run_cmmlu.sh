model_path=path/to/chinese_llama2_or_alpaca2
output_path=path/to/your_output_dir

#cd scripts/cmmlu
python eval.py \
    --model_path ${model_path} \
    --few_shot False \
    --with_prompt True \
    --constrained_decoding True \
    --output_dir ${output_path} \
    --input_dir data