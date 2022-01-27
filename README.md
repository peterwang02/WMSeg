# WMsegmentation

INFO:
    This is modified/extended code of the 2020ACL paper "Improving Chinese Word Segmentation with Wordhood Memory Networks". The author's implementation, which is what this code is based on, can be found here https://github.com/SVAIGBA/WMSeg. Most of the code is directly from the repository given, however, with modifications to almost every file and function. The most significant changes are: 
    1. removed support for ZEN and SOFTMAX, the encoder-decoder combination used for all runs is BERT-CRF, this is mainly shown in wmseg_main.py, wmseg_helper.py, and wmseg_model.py; 
    2. rewrote/extended getgram2id from wmseg_helper.py to make it produce three different wordhood measurement structures; 
    3. heavily modifed/extended the WORDKVNN neural network module from wmseg_model.py, and corresponding changes in other functions in this file as well; 
    4. changed reporting of results from printing directly to stdout to printing to "testEval.txt" (in wmseg_main.py).
    5. modified very little code in pytorch_pretrained_bert to make it compatible with the newest pytorch version

Running Experiments:
    *Description of these args can be found in wmseg_main.py, but most should be self-explanatory*

    to train and evaluate: python wmseg_main.py --do_train --train_data_path=./data/msr/train.tsv --eval_data_path=./data/msr/test.tsv --bert_model=bert-base-chinese --max_seq_length=300 --max_ngram_size=300 --train_batch_size=8 --eval_batch_size=8 --num_train_epochs=10 --warmup_proportion=0.1 --learning_rate=5e-5 --model_name=model_name --use_memory

    to evaluate only: python wmseg_main.py --do_test --eval_data_path=./path/to/testdata --eval_model=./models/model_name/model.pt

Segmentation on a file:
    python wmseg_main.py --do_predict --input_file=./path/to/input --output_file=./path/to/output --eval_model=./models/model_name/model.pt
    
Results: 
    Training and Testing results on SIGHAN2005 are provided in the results folder.
    
Dataset files:
    Small dataset file available in /sample_data/. As they are quite small (<10kb), they are included in this repo
    You can run the data_preprocessing.py to retrieve SIGHAN2005 datasets and reproduce the results provided.


