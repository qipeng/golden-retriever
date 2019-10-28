echo "Downloading prepared data..."

cd prepared_data
wget https://nlp.stanford.edu/projects/golden-retriever/prepared_data.zip

echo "Extracting files..."
unzip prepared_data.zip
rm prepared_data.zip
echo "Done!"
