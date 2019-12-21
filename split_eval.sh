#[range(0, 741), range(741, 1482), range(1482, 2223), range(2223, 2964), range(2964, 3705), range(3705, 4445), range(4445, 5185), range(5185, 5925), range(5925, 6665), range(6665, 7405)]

# Args
#N_SPLITS=3
#SET="dev"
#FULLWIKI=false
#INFER_BATCHSIZE=24
N_SPLITS=$1
SET=$2
FULLWIKI=$3
INFER_BATCHSIZE=$4
BERT_BATCHSIZE=$5

if $FULLWIKI; then
    DIR="bert_${SET}_fullwiki_submit_${N_SPLITS}splits"
else
    DIR="bert_${SET}_distractor_submit_${N_SPLITS}splits"
fi

source activate pt3
if $FULLWIKI; then
    N_DATA=$(python get_record_infos.py $SET $DIR --fullwiki )
else
    N_DATA=$(python get_record_infos.py $SET $DIR)
fi
source deactivate

echo "[N splits : $N_SPLITS]"
echo "[Set : $SET]"
echo "[Fullwiki : $FULLWIKI]"
echo "[Dataset length : $N_DATA]"
echo "[Inference batch size : $INFER_BATCHSIZE]"
echo "[Saving to : $DIR]"

# Get id spans for splits
start_ids=($(python calc_splits_batch.py $N_DATA $N_SPLITS $INFER_BATCHSIZE start))
end_ids=($(python calc_splits_batch.py $N_DATA $N_SPLITS $INFER_BATCHSIZE end))
#echo ${start_ids[@]}
#echo ${end_ids[@]}

for ((i=0;i<${#start_ids[@]};++i)); do
    printf "Split %s: Start %s End %s\n" "${i}" "${start_ids[i]}" "${end_ids[i]}"
    source activate bert
    if $FULLWIKI; then
        python process_bert_hdf5_split.py --data_split ${SET} --layer_pooling -2 --n_proc 12 --window_pooling avg --wordpiece_pooling sum --batch_size ${BERT_BATCHSIZE} --save_dir ${DIR} --data_start ${start_ids[i]} --data_end ${end_ids[i]} --fullwiki
    else
        python process_bert_hdf5_split.py --data_split ${SET} --layer_pooling -2 --n_proc 12 --window_pooling avg --wordpiece_pooling sum --batch_size ${BERT_BATCHSIZE} --save_dir ${DIR} --data_start ${start_ids[i]} --data_end ${end_ids[i]}
    fi
    source deactivate
    source activate pt3
    if $FULLWIKI; then
        CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_split ${SET} --para_limit 2250 --batch_size ${INFER_BATCHSIZE} --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0 --save model --prediction_file "${DIR}/split${i}_pred.json" --reasoning_module MACv2 --reasoning_steps 3 --sp_thresh 0.5 --bert --bert_with_glove --bert_dir ${DIR} --pointing --fullwiki
    else
        CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_split ${SET} --para_limit 2250 --batch_size ${INFER_BATCHSIZE} --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0 --save model --prediction_file "${DIR}/split${i}_pred.json" --reasoning_module MACv2 --reasoning_steps 3 --sp_thresh 0.5 --bert --bert_with_glove --bert_dir ${DIR} --pointing
    fi
    source deactivate
done
python merge_splits.py ${DIR} ${N_SPLITS}
