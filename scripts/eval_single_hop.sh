#! /bin/bash

# Instructions:
# 1. clone the DrQA repo
#  - run ./download.sh in that repo and ./install_corenlp.sh
#  - Download Glove embeddings http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
# 2. call this script from the deep-retriever directory
# 3. conda activate hotpot
set -e  # stop script if any command fails

# Script parameters for users (feel free to edit):
OUTDIR="hotpotqa_dev_singlehop_eval"  # suggested convention is "[dataset_name]_eval"
QUESTION_FILE="data/hotpotqa/hotpot_dev_distractor_v1.json"
#QA_MODEL_NAME="HOTPOT-20190514-073647" # old model trained with official hotpot data
QA_MODEL_NAME="HOTPOT-20190520-075452"

DRQA_DIR="DrQA"
EMBED_FILE="${DRQA_DIR}/data/embeddings/glove.840B.300d.txt"
EVAL_FILE="data/hotpotqa/hotpot_dev_distractor_v1.json"
RECOMPUTE_ALL=false  # change to `true` to force recompute everything

# set -x  # UNCOMMENT this line for debugging output

# Change code below this line at your own risk!
###########################################################################################

echo "Placing temporary evaluation files in: $OUTDIR"
mkdir -p $OUTDIR

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

if $RECOMPUTE_ALL || [ ! -f data/hotpotqa/hotpot_dev_single_hop.json ]; then
  python -m scripts.build_single_hop_qa_data dev
fi

WD=`pwd`
cd /u/scr/pengqi/HotPotQA
if $RECOMPUTE_ALL || [ ! -f $WD/$OUTDIR/golden.json ]; then
  python main.py --mode prepro --data_file $WD/data/hotpotqa/hotpot_dev_single_hop.json --para_limit 2250 --data_split test --fullwiki
  python main.py --mode test --data_split test --save ${QA_MODEL_NAME} --prediction_file $WD/$OUTDIR/golden.json --sp_lambda 10.0 --fullwiki --hidden 128 --batch_size 16
else
  echo 'Using existing prediction file:'
fi
ls $WD/$OUTDIR/golden.json

if [ -f $WD/${EVAL_FILE} ]; then
    python hotpot_evaluate_v1.py $WD/$OUTDIR/golden.json $WD/${EVAL_FILE} | python $WD/scripts/format_result.py
fi

echo "Done! Final results in: $OUTDIR/golden.json"

