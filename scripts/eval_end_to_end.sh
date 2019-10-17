#! /bin/bash

# Instructions:
# - make sure you have already run `setup.sh` and are using the correct python environment
# - call this script from the root directory of this project
# - please feel free to modify any of the script inputs below

set -e  # stop script if any command fails

# Script parameters for users (feel free to edit):
OUTDIR="outdir"  # suggested convention is "[dataset_name]_eval"
QUESTION_FILE="data/hotpotqa/hotpot_dev_distractor_v1.json"
HOP1_MODEL_NAME="hop1"
HOP2_MODEL_NAME="hop2"
QA_MODEL_NAME="QAModel"

DRQA_DIR="DrQA"
EMBED_FILE="${DRQA_DIR}/data/embeddings/glove.840B.300d.txt"
RECOMPUTE_ALL=false  # change to `true` to force recompute everything
NUM_DRQA_WORKERS=16
BIDAFPP_DIR="BiDAFpp"

# Toggle these settings to experiment with oracle queries in the pipeline
USE_HOP1_ORACLE=false
USE_HOP2_ORACLE=false

# set -x  # UNCOMMENT this line for debugging output

# Change code below this line at your own risk!
###########################################################################################

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

# use DrQA's version of corenlp
export CLASSPATH="`realpath ${DRQA_DIR}`/data/corenlp/*:$CLASSPATH:."
export CORENLP_HOME=`realpath stanford-corenlp-full-2018-10-05`

HOP1_MODEL_FILE="models/$HOP1_MODEL_NAME.mdl"
HOP2_MODEL_FILE="models/$HOP2_MODEL_NAME.mdl"
HOP1_LABEL="data/hop1/hotpot_hop1_dev.json"
HOP2_LABEL="data/hop2/hotpot_hop2_dev.json"

CLASSPATH=${DRQA_DIR}/data/corenlp/*

if [ ! -d ${DRQA_DIR} ]
then
  echo "Make sure you've cloned the DrQA repo in ${DRQA_DIR}"
  exit 1
fi

if [ ! -f $EMBED_FILE ]
then
  echo "Download the Glove embeddings and place them: $EMBED_FILE"
  echo "http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip"
  exit 1
fi

if [ ! -f $HOP1_MODEL_FILE ]
then
  echo "Make sure your HOP1 model file exists: $HOP1_MODEL_FILE"
  exit 1
fi

if [ ! -f $HOP2_MODEL_FILE ]
then
  echo "Make sure your HOP2 model file exists: $HOP2_MODEL_FILE"
  exit 1
fi

if [ ! -f $QUESTION_FILE ]
then
  echo "Make sure your Question file exists: $QUESTION_FILE"
  exit 1
fi

echo "Placing temporary evaluation files in: $OUTDIR"
mkdir -p $OUTDIR

if $RECOMPUTE_ALL || [ ! -f $OUTDIR/hop1_squadified.json ]
then
  python -m scripts.e_to_e_helpers.squadify_questions $QUESTION_FILE $OUTDIR/hop1_squadified.json
fi

HOP1_PREDICTIONS="$OUTDIR/hop1_squadified-$HOP1_MODEL_NAME.preds"
if $RECOMPUTE_ALL || [ ! -f $HOP1_PREDICTIONS ]
then
  echo "Generating hop1 predictions..."
  if $USE_HOP1_ORACLE; then
    python scripts/query_labels_to_pred.py $HOP1_LABEL $HOP1_PREDICTIONS
  else
    python ${DRQA_DIR}/scripts/reader/predict.py $OUTDIR/hop1_squadified.json --out-dir $OUTDIR --num-workers $NUM_DRQA_WORKERS --embedding-file $EMBED_FILE --model $HOP1_MODEL_FILE
  fi
fi

echo "Hop1 predicted labels:"
ls -la $HOP1_PREDICTIONS

echo "Trying to connect to ES at localhost:9200..."
if ! curl -s -I localhost:9200 > /dev/null;
then
  echo 'running "sh scripts/launch_elasticsearch_6.7.sh"'
  sh scripts/launch_elasticsearch_6.7.sh
  while ! curl -I localhost:9200;
  do
    sleep 2;
  done
fi
echo "ES is up and running"

if [ ! -f $OUTDIR/hop2_input.json ] \
    || [ ! -f $OUTDIR/SQuAD_hop2_input.json ] \
    || $RECOMPUTE_ALL
then
  echo "Creating input for hop2 query prediction"
  python -m scripts.e_to_e_helpers.merge_with_es \
    $HOP1_PREDICTIONS \
    $QUESTION_FILE \
    $OUTDIR/hop2_input.json

  python -m scripts.preprocess_hop2 $OUTDIR hop2_input.json
  echo "Created Hop2 SQuAD-formatted input:"
else
  echo 'Using existing Hop2 SQuAD-formatted input file:'
fi
ls -la $OUTDIR/SQuAD_hop2_input.json

HOP2_PREDICTIONS="$OUTDIR/SQuAD_hop2_input-$HOP2_MODEL_NAME.preds"
if $RECOMPUTE_ALL || [ ! -f $HOP2_PREDICTIONS ]
then
  if $USE_HOP2_ORACLE; then
    python scripts/query_labels_to_pred.py $HOP2_LABEL $HOP2_PREDICTIONS
  else
    python ${DRQA_DIR}/scripts/reader/predict.py $OUTDIR/SQuAD_hop2_input.json --out-dir $OUTDIR --num-workers $NUM_DRQA_WORKERS --embedding-file $EMBED_FILE --model $HOP2_MODEL_FILE
  fi
fi

echo "Hop2 predictions:"
ls -la $HOP2_PREDICTIONS

if $RECOMPUTE_ALL || [ ! -f $OUTDIR/hop2_output.json ]
then
  echo "Querying ES with hop2 predictions"
  python -m scripts.e_to_e_helpers.merge_with_es \
    $HOP2_PREDICTIONS \
    $QUESTION_FILE \
    $OUTDIR/hop2_output.json
  echo "Created Hop2 output:"
else
  echo "Using existing Hop2 output:"
fi
ls -la $OUTDIR/hop2_output.json

if $RECOMPUTE_ALL || [ ! -f $OUTDIR/qa_input.json ]; then
  python -m scripts.e_to_e_helpers.merge_hops_results \
    $OUTDIR/hop2_input.json \
    $OUTDIR/hop2_output.json \
    $OUTDIR/qa_input.json \
    --include_queries \
    --num_each 5
  echo "Created QA output:"
else
  echo "Using existing QA output:"
fi
ls -la $OUTDIR/qa_input.json

WD=`pwd`
if $RECOMPUTE_ALL || [ ! -f $WD/$OUTDIR/golden.json ]; then
  pushd $BIDAFPP_DIR
  python main.py --mode prepro --data_file $WD/$OUTDIR/qa_input.json --para_limit 2250 --data_split test --fullwiki
  python main.py --mode test --data_split test --save ${QA_MODEL_NAME} --prediction_file $WD/$OUTDIR/golden.json --sp_threshold .33 --sp_lambda 10.0 --fullwiki --hidden 128 --batch_size 16
  popd
fi
ls $WD/$OUTDIR/golden.json

if [ -f $WD/${QUESTION_FILE} ]; then
    cd $BIDAFPP_DIR
    python hotpot_evaluate_v1.py $WD/$OUTDIR/golden.json $WD/${QUESTION_FILE} | python $WD/scripts/format_result.py
fi
cd $WD

echo "Done! Final results in: $OUTDIR/golden.json"

