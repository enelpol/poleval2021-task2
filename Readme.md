# PolEval 2021 Task 2: Evaluation of translation quality assessment metrics

## Blind

### Results

|                  | test-B Pearson |
|------------------|----------------|
| M1               | 0.5076         |
| B1               | **0.5138**  |
|||
| Darek Kłeczek    | 0.4840         |
| Artur Nowakowski | 0.4793         |

### Instructions

```shell
git clone https://github.com/poleval/2021-quality-estimation-blind
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

```shell
MODEL="enelpol/poleval2021-task2"
python3 predict_blind.py --model_name $MODEL 2021-quality-estimation-blind/test-B/in.tsv > output.tsv
```