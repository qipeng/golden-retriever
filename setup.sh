set -e

python -c 'import sys; print(sys.version_info[:])'
echo "Please make sure you are running python version 3.6.X"

echo "Installing required Python packages..."
pip install -r requirements.txt

echo "Setting up DrQA..."
pushd DrQA
pip install -r requirements.txt
python setup.py develop
./install_corenlp.sh
popd

wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
mkdir DrQA/data/embeddings
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
mv glove.840B.300d.txt DrQA/data/embeddings/glove.840B.300d.txt

echo "Downloading models..."
bash scripts/download_golden_retriever_models.sh

echo "Getting HotpotQA dataset..."
bash scripts/download_hotpotqa.sh

echo "Downloading Elasticsearch..."
bash scripts/download_elastic_6.7.sh

echo "NOTE: we set jvm options -Xms and -Xmx for Elasticsearch to be 4GB"
echo "We suggest you set them as large as possible in: elasticsearch-6.7.0/config/jvm.options"
cp search/jvm.options elasticsearch-6.7.0/config/jvm.options

echo "Downloading wikipedia source documents..."
bash scripts/download_processed_wiki.sh

echo "Running Elasticsearch and indexing Wikipedia documents..."
bash scripts/launch_elasticsearch_6.7.sh
python -m scripts.index_processed_wiki

echo "Download CoreNLP..."
bash scripts/download_corenlp.sh

echo "Setup BiDAF++..."
pip install spacy
pushd BiDAFpp
./download.sh
popd

