#!/bin/bash

project=(tile_0_563)
epochs=2
psize=(24)
cell=(8)
input=(bands)
experiment=(5_azure)
fold=(0)
batchsize=(8)
trainon=(2002)
optimizertype=(nadam)
LR=(0.0001)
reference=(MCD12Q1v6stable01to03_LCProp2)
step=(training)

mkdir -p "_logs/${project}/models/$experiment"
echo "Processing $input and fold: $fold and reference: $reference and optimiser $optimizertype"
logfname="_logs/${project}/models/$experiment/${input}_fold${fold}_${reference}_${optimizertype}.log"
python runtrain.py \
    --step=$step \
    --train_on $trainon \
    --fold=$fold \
    --epochs=$epochs \
    --experiment $input \
    --reference $reference \
    --batchsize $batchsize \
    --optimizertype $optimizertype \
    --convrnn_filters $cell \
    --learning_rate $LR \
    --pix250m $psize > $logfname 2>&1