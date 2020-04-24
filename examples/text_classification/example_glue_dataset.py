import argparse
import mxnet as mx
import autogluon as ag
from autogluon import TextClassification as task

parser = argparse.ArgumentParser(
    description='BERT fine-tune examples for classification/regression tasks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--task_name',
    type=str,
    choices=['MRPC', 'QNLI', 'RTE', 'STS-B', 'CoLA',
             'MNLI', 'WNLI', 'SST', 'XNLI', 'LCQMC', 'ChnSentiCorp'],
    help='The name of the task to fine-tune. Choices include MRPC, QQP, '
         'QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST.')
parser.add_argument('--optimizer', type=str, default='bertadam',
                    help='The optimizer to be used for training')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size. Number of examples per gpu in a minibatch.')
parser.add_argument(
    '--dev_batch_size',
    type=int,
    default=8,
    help='Batch size for dev set and test set')
parser.add_argument(
    '--lr',
    type=float,
    default=3e-5,
    help='Initial learning rate')
parser.add_argument(
    '--epsilon',
    type=float,
    default=1e-6,
    help='Small value to avoid division by 0'
)
parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.1,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='report interval')
parser.add_argument(
    '--max_len',
    type=int,
    default=128,
    help='Maximum length of the sentence pairs')
parser.add_argument(
    '--seed', type=int, default=2, help='Random seed')
parser.add_argument(
    '--accumulate',
    type=int,
    default=None,
    help='The number of batches for gradients accumulation to simulate large batch size. '
         'Default is None')
parser.add_argument(
    '--bert_model',
    type=str,
    default='bert_12_768_12',
    choices=['bert_12_768_12', 'bert_24_1024_16', 'roberta_12_768_12', 'roberta_24_1024_16'],
    help='The name of pre-trained BERT model to fine-tune')
parser.add_argument(
    '--bert_dataset',
    type=str,
    default='book_corpus_wiki_en_uncased',
    choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
             'openwebtext_book_corpus_wiki_en_uncased', 'wiki_multilingual_uncased',
             'wiki_multilingual_cased', 'wiki_cn_cased',
             'openwebtext_ccnews_stories_books_cased'],
    help='The dataset BERT pre-trained with.')
parser.add_argument(
    '--output_dir',
    type=str,
    default='checkpoint/exp1.ag',
    help='The output directory where the model params will be written.')
parser.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float16'],
    help='The data type for training.')
parser.add_argument(
    '--num_trials',
    type=int,
    default=50,
    help='number of trials. ')

args = parser.parse_args()

dataset = task.Dataset(name=args.task_name)
predictor = task.fit(dataset,
                     net=ag.Categorical(args.bert_model),
                     lr=ag.Real(args.lr / 10, args.lr * 10, log=True),
                     warmup_ratio=ag.Real(args.warmup_ratio / 10, args.warmup_ratio * 10, log=True),
                     log_interval=args.log_interval,
                     seed=args.seed,
                     batch_size=args.batch_size,
                     dev_batch_size=args.dev_batch_size,
                     max_len=args.max_len,
                     dtype=args.dtype,
                     epochs=args.epochs,
                     epsilon=ag.Real(args.epsilon / 10, args.epsilon * 10, log=True),
                     accumulate=args.accumulate,
                     checkpoint=args.output_dir,
                     num_trials=args.num_trials)
print('Val reward:')
print(predictor.results['best_reward'])
test_reward = predictor.evaluate(dataset)
print('Test reward:')
print(test_reward)
print('The best configuration is:')
print(predictor.results['best_config'])
