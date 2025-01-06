export OPENAI_API_KEY="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"

for i in {1..10}
do
    python ../geval.py \
        --input_path '../results/0102/multicritic-ddxplus-biomistral-7b-biomistral-7b-biomistral-7b.json' \
        --prompt '../prompts/cds/explainability.txt' \
        --save_dir "../outputs/ddxplus-explainability/biomistral-7b/run_$i"
done
