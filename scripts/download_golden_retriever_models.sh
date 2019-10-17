mkdir -p models
pushd models

echo "Downloading query generators..."
wget https://nlp.stanford.edu/projects/golden-retriever/hop1.mdl
wget https://nlp.stanford.edu/projects/golden-retriever/hop2.mdl

echo "Downloading QA model..."
cd ../BiDAFpp
wget https://nlp.stanford.edu/projects/golden-retriever/QAModel.zip
unzip QAModel.zip
rm QAModel.zip
wget https://nlp.stanford.edu/projects/golden-retriever/jsons.zip
unzip jsons.zip
rm jsons.zip
popd
echo "Done!"
