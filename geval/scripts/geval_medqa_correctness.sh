export OPENAI_API_KEY="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"

for i in {1..10}
do
    python ../geval.py \
        --input_path '../results/0102/multicritic-medqa-meditron-70b-meditron-70b-meditron-70b.json' \
        --prompt '../prompts/kqa/correctness.txt' \
        --save_dir "../outputs/medqa-correctness/meditron-70b/run_$i"
done
