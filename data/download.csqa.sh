mkdir data/CSQA
wget https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl -O data/CSQA/train.jsonl
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl -O data/CSQA/dev.jsonl
wget https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl -O data/CSQA/test.jsonl
