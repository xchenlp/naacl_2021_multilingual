#!/bin/bash

runs="1 2 3 4 5 6 7 8 9 10"

for i in $runs
do
    python stackoverflow_classification.py --encoder  distil-m-bert-so-multi distil-m-bert m-bert --classifier multi_att --embedding_type word --data stackoverflow --language eng esp por rus jap --out_path naacl_eval/output_encoders_$i

    python stackoverflow_classification.py --encoder  distil-m-bert-so-multi distil-m-bert m-bert --classifier multi_att --embedding_type word --data stackoverflow --language esp-xlt por-xlt rus-xlt jap-xlt --out_path naacl_eval/xlt/output_encoders_$i

    echo "FINISHED BATCH $i"
done
