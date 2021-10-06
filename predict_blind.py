import os
import random
import time
from argparse import ArgumentParser

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

from scipy.stats import pearsonr, spearmanr


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]

def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def read_blind(path, normalize=True):
    directory = path
    data = []
    for text, score in zip(open(f"{directory}/in.tsv"), open(f"{directory}/expected.tsv")):
        score = float(score)
        if score > 5: continue
        
        text = text.split('\t')[0].strip()
        if normalize:
            score = (score - 1) / 4
        # print(text, score)
        data.append([text, score])
    return data


def read_blind_test(path):
    data = []
    for text in open(path):
        text = text.split('\t')[0].strip()

        data.append(text)
    return data

if __name__ == '__main__':
    parser = ArgumentParser(description='Train model')
    parser.add_argument('in_path',  help='Path to directory with in and expected TSV')
    # parser.add_argument('--test', help='Path to test TSV')
    parser.add_argument('--model_type', default="auto", help='Type of the model', )
    parser.add_argument('--model_name', default="allegro/herbert-base-cased", help='Name of the model', )
    parser.add_argument('--wandb_project', default='regression', help='Project name in wandb')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--epochs', default=4, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='mini batch size')
    parser.add_argument('--early_stopping_metric', default='pearson', help='early_stopping_metric')
    parser.add_argument('--acc', default=1, type=int, help='gradient_accumulation_steps')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning rate')
    # parser.add_argument('--eps', default=1e-8, type=float, help='adam eps')
    parser.add_argument('--gradient', default=1.0, type=float, help='gradient')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay (L2 penalty)')
    parser.add_argument('--max_seq_length', default=256, type=int, help='max_seq_length')
    # parser.add_argument('--weights', nargs='+', default=None, type=float, help='weights of 1 to balance')
    # parser.add_argument('--labels_list', nargs='+', default=['0', '1'],
    #                     help='list of label names; in binary problem the second label is positive')
    # parser.add_argument('--labels_path', default=None, help='path to file with labels')
    parser.add_argument('--evaluate_steps', default=20, type=int, help='evaluate every x steps')
    # TODO outputdir

    cache_dir = os.getenv('SCRATCH', '.') + '/cache/'
    args = parser.parse_args()

    # print(args)
    train_data=read_blind_test(args.in_path)



    # train_df = pd.DataFrame(train_data)
    # 
    # if len(train_df.columns)==2:
    #     train_df.columns = ["text", "labels"]
    # elif len(train_df.columns)==3:
    #     train_df.columns = ["text_a","text_b", "labels"]


    model_args = ClassificationArgs()
    model_args.num_train_epochs = args.epochs
    model_args.regression = True

    model_args.train_batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size

    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.wandb_project = args.wandb_project
    model_args.manual_seed = args.seed
    model_args.early_stopping_metric = args.early_stopping_metric
    model_args.gradient_accumulation_steps = args.acc
    model_args.learning_rate = args.learning_rate
    model_args.max_grad_norm = args.gradient
    model_args.weight_decay = args.weight_decay
    model_args.max_seq_length = args.max_seq_length
    model_args.early_stopping_metric_minimize = False
    model_args.evaluate_during_training = True
    model_args.warmup_steps = 0
    model_args.warmup_ratio = 0.06
    model_args.evaluate_during_training_steps = args.evaluate_steps
    model_args.save_eval_checkpoints = False
    model_args.logging_steps = args.evaluate_steps

    model_args.save_steps = -1
    model_args.save_model_every_epoch = False

    time_of_run = time.time()
    output_dir = 'model_bin_' + args.model_name.replace('/', '_') + '_' + str(
        time_of_run)
    best_model_dir = output_dir + "/best_model/"

    model_args.output_dir = output_dir
    model_args.best_model_dir = best_model_dir

    # Create a ClassificationModel
    model = ClassificationModel(args.model_type, args.model_name,
                                num_labels=1,
                                args=model_args
                                )

    preds, model_outputs = model.predict(train_data)

    # print(preds)
    for pred in preds:
        print(pred)