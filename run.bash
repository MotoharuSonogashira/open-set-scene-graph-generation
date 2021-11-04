#!/bin/bash

set -eu

: ${DRY:=0}
: ${GPUS:=0} # 0 means single gpu mode
: ${TEST:=0}
: ${OPEN:=0}
: ${METHOD:=faster_rcnn}
: ${THRESHOLD:=}
if ((TEST)) && [[ -n $THRESHOLD ]]; then
    TEST_UNKNOWN=1
fi
: ${CONFIG:=X_101_32_8_FPN_1x}
: ${VAL:=0} # nonzero for hyperparameter validation (inner learning)
: ${TEST_AFTER_TRAIN:=0} # enable testing after training
: ${MODEL_FINAL:=0} # use model_final.pth instead of last_checkpoint
                              # in testing
: ${ROOT:=output}     # training root directory
: ${TEST_ROOT:=$ROOT} # testing  root directory
: ${MAX_ITER:=} # maximum number of iterations in training/testing
: ${SKIP:=0} # skip training/testing if it is already done
: ${WAIT:=0} # wait for the weight file to be made
: ${NO_CACHE:=0} # do not use cached predictions in testing

: ${TIMEOUT:=100000}
export TIMEOUT
     # default value of init_process_group (1800) succeeds for inner test (10k
     # images) but fails by timeout error for outer test (30k images)

function p() {
    printf -- "$*\n"
}

function @() {
    p "$@"
    ((DRY)) && return || :
    "$@"
}


# executable
exe=(python3 $( ((GPUS)) && p "-m torch.distributed.launch --nproc_per_node=$GPUS --max_restarts=0" || :))
    # max_restarts prevents hanging in worker group restart after exception is thrown
cfg=(DTYPE float16 TEST.IMS_PER_BATCH $( ((GPUS)) && p $GPUS || p 1))
    # IMS_PER_BATCH must be divisible by the num. GPUS

# data
data_name=$( ((OPEN)) && p open || p original)$( ( ((OPEN)) && ( ((VAL)) && p -inner || p -outer)) || :)
dataset=$( ((OPEN)) && p VG_stanford_filtered_open || p VG_stanford_filtered_with_attribute)
for s in train val test; do
    suffix=
    for t in unknown missing; do
        var="${s^^}_${t^^}" # 'TEST_UNKNOWN' etc
        if ((${!var:-0})); then
            suffix+=_${t}
        fi
    done
    cfg+=(DATASETS.${s^^} "('${dataset}_${s}${suffix}',)")
    if [[ $s == test ]]; then
        test_dataset=${dataset}_${s}${suffix}
    fi
done
if ((VAL)); then # inner
    cfg+=(DATASETS.INPUT_VAL test)
        # input-split valid data are used as test
else # outer
    cfg+=(DATASETS.INPUT_VAL train)
        # input-split valid data are merged to train
fi

# method
declare -A predictors=([dummy]=DummyPredictor [motif]=MotifPredictor [imp]=IMPPredictor [vctree]=VCTreePredictor)
declare -A contexts=([motif]=motifs [vtranse]=vtranse [vctree]=vctree)
method_name=${METHOD_NAME:-$METHOD}
test_method_name=${TEST_METHOD_NAME:-$method_name}
model=
if [[ $METHOD == faster_rcnn ]]; then
    cfg+=(MODEL.RELATION_ON False)
else
    if ! [[ $METHOD =~ ^([[:alnum:]_]+)?$ ]]; then
        echo "Error: invalid method: '$METHOD'" >&2
        exit 1
    fi
    train_method=${BASH_REMATCH[1]}
    freeze_freq=0
    bias=
    if [[ $train_method == freq ]]; then # dummy model w/ freq
        model=dummy
        bias=1
    else
        model=$train_method
    fi

    cfg+=(MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictors[$model]})
    if (($freeze_freq)); then
        cfg+=(MODEL.FREQ.TRAINABLE False)
    fi
    if [[ -n $bias ]]; then
        cfg+=(MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS $( ((bias)) && p True || p False))
    fi
fi
if [[ -n $THRESHOLD ]]; then
    cfg+=(MODEL.UNKNOWN_THRESHOLD $THRESHOLD)
    test_method_name=${TEST_METHOD_NAME:-${test_method_name}-$THRESHOLD}
    # use same training directory regardless of UNKNOWN_THRESHOLD
fi
test_method_name=${test_method_name}${PROTOCOL:+-$PROTOCOL}

# paths
detector_dir=$ROOT/$data_name/faster_rcnn
dir=$ROOT/$data_name/$method_name
test_dir=$TEST_ROOT/$data_name/$test_method_name
cfg+=(GLOVE_DIR $HOME/glove OUTPUT_DIR $dir)
if ! ((TEST)); then # training
    log=$dir/log.txt
    if ((SKIP)) && [[ -e $log ]] && grep -q 'Total training time' "$log"; then
        exit 0
    fi
else # testing
    cfg+=(TEST_OUTPUT_DIR $test_dir)
    if ! ((NO_CACHE)); then
        cfg+=(TEST.ALLOW_LOAD_FROM_CACHE True)
    fi
    if ((SKIP)) && [[ -e $test_dir/inference/$test_dataset/result_dict.pytorch ]]; then
        exit 0
    fi
fi

# mode
declare -A protocols=([predcls]="MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True" [sgcls]="MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False" [sgdet]="MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False")
weight= # no weight is loaded if empty
if [[ $METHOD == faster_rcnn ]]; then
    config=configs/e2e_relation_detector_${CONFIG}.yaml
    if ! ((TEST)); then # training
        script=tools/detector_pretrain_net.py
        cfg+=(SOLVER.IMS_PER_BATCH 8 SOLVER.STEPS "(30000, 45000)")
    else # testing
        script=tools/detector_pretest_net.py
    fi
else
    config=configs/e2e_relation_${CONFIG}.yaml
    if ! ((TEST)); then # training
        script=tools/relation_train_net.py
        cfg+=(SOLVER.IMS_PER_BATCH 12)
        weight=$detector_dir/model_final.pth
    else # testing
        script=tools/relation_test_net.py
    fi
    cfg+=(${protocols[${PROTOCOL:-sgdet}]})
fi
if ! ((TEST)); then # training
    cfg+=(SOLVER.MAX_ITER ${MAX_ITER:-$([[ $model == dummy ]] && p 0 || p 50000 )} SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.PRE_VAL False)
        # dummy model does not need training iteration (only bias cache is
        # saved)
else # testing
    if ((MODEL_FINAL)); then
        weight=$dir/model_final.pth
    fi
    if [[ -n $MAX_ITER ]]; then
        cfg+=(DATASETS.NUM_IM $MAX_ITER)
    fi
fi
if [[ -n $weight ]]; then
    if [[ ! -e $weight ]] && ! ((DRY)); then # weight file does not exist
        if ((WAIT)); then
            while [[ ! -e $weight ]]; do
                sleep 60
            done
            sleep 600
                # wait several more minutes after file is created since saving
                # may be in progress
        else
            echo "Error: weight not found: '$weight'." >&2
            exit
        fi
    fi
    cfg+=(MODEL.PRETRAINED_DETECTOR_CKPT $weight)
fi
opts=(--config-file $config)
if ! ((TEST)) && ! ((TEST_AFTER_TRAIN)); then
    opts+=(--skip-test)
fi

@ "${exe[@]}" "$script" "${opts[@]}" "${cfg[@]}" "$@"
