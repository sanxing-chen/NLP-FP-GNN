# Context Dependent Text-to-SQL Semantic Parsing

This's based on a fork from the author implementation of the ACL 2019 paper: [Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing](https://arxiv.org/abs/1905.06241).

## Install & Configure

1. Install pytorch version 1.2.0 that fits your CUDA version 

(Different from the origial implementation, this repository use the latest version of pytorch and other packages.)

2. Install the rest of required packages
    ```
    pip install -r requirements.txt
    ```
    
3. Run this command to install NLTK punkt.
```
python -c "import nltk; nltk.download('punkt')"
```

4. Download the dataset from the [official Spider dataset website](https://yale-lily.github.io/spider)

5. Edit the config file `train_configs/defaults.jsonnet` to update the location of the dataset:
```
local dataset_path = "dataset/";
```

6. **Before preprocessing the dataset, modify [two](https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/data/fields/knowledge_graph_field.py#L99) [lines](https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/data/fields/knowledge_graph_field.py#L109) in allennlp lib, to replace `self._tokenizer` with `_tokenizer`. This change greatly reduces the size of cache data and memory usage.** Also, change the number of processes in `dataset_readers/spider.py` according to your machine setting.

## Training

1. Use the following AllenNLP command to train:
```
allennlp train train_configs/defaults.jsonnet -s experiments/name_of_experiment \
--include-package dataset_readers.spider \ 
--include-package models.semantic_parsing.spider_parser
``` 

First time loading of the dataset might take a while (a few hours) since the model first loads values from tables and calculates similarity features with the relevant question. It will then be cached for subsequent runs.

You should get results similar to the following (the `sql_match` is the one measured in the official evaluation test):
```
  "best_validation__match/exact_match": 0.3715686274509804,
  "best_validation_sql_match": 0.47549019607843135,
  "best_validation__others/action_similarity": 0.5731271471206189,
  "best_validation__match/match_single": 0.6254612546125461,
  "best_validation__match/match_hard": 0.3054393305439331,
  "best_validation_beam_hit": 0.6070588235294118,
  "best_validation_loss": 7.383035182952881
  "best_epoch": 32
```

Note that the hyper-parameters used in `defaults.jsonnet` are different than those mentioned in the paper
(most importantly, 3 timesteps are used instead of 2), thanks to the [following contribution from @wlhgtc](https://github.com/benbogin/spider-schema-gnn/pull/13).
The original training config file is still available in `train_configs/paper_Defaults.jsonnet`.

## Inference

Use the following AllenNLP command to output a file with the predicted queries:

```
allennlp predict experiments/name_of_experiment dataset/dev.json \
--predictor spider \
--use-dataset-reader \
--cuda-device=0 \
--output-file experiments/name_of_experiment/prediction.sql \
--silent \
--include-package models.semantic_parsing.spider_parser \
--include-package dataset_readers.spider \
--include-package predictors.spider_predictor \
--weights-file experiments/name_of_experiment/best.th \
-o "{\"dataset_reader\":{\"keep_if_unparsable\":true}}"
```

## Debug

Refer to [AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/using_a_debugger.md), use `run.py` for debugging.