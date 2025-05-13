#!/bin/bash

set -x

source /ocean/projects/cis220039p/mdt2/ylin23/miniconda3/bin/activate
conda activate 3dvista

export SSL_CERT_FILE=/ocean/projects/cis220039p/mdt2/ylin23/cacert.pem

export RUN_DIR=/ocean/projects/cis220039p/mdt2/ylin23/scanrefer/feats/3D-VisTA

cd $RUN_DIR

python3 run.py --config project/vista_reproduce/pretrain_2d.yml