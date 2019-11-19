import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--maintrain', action='store_true', default=False)
args = parser.parse_args()
################################################################################
# general params
fs='16000'
fftsise='512'
framesize='128'  # 1.028[sec]
num_hidden='256'
dim_embedding='128'
num_embedding='512'
batchsize='128'
epoch='100'
beta='0.25'
learnrate='3e-4'
checkpoint='5'
################################################################################
# run experiment
if args.pretrain:
    subprocess.run(['python',
                    'pretrain_timit.py',
                    fftsise,
                    framesize,
                    num_hidden,
                    dim_embedding,
                    num_embedding,
                    beta,
                    batchsize,
                    epoch,
                    learnrate,
                    checkpoint])
if args.maintrain:
    subprocess.run(['python',
                    'train_timit.py',
                    fftsise,
                    framesize,
                    num_hidden,
                    dim_embedding,
                    num_embedding,
                    batchsize,
                    '200',
                    '1e-4',
                    checkpoint,
                    'model/speech_model.pth'])
