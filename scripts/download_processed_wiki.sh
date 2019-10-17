# download the wiki dump file
mkdir -p data
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -O data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
# verify that we have the whole thing
unameOut="$(uname -s)"
case "${unameOut}" in
    Darwin*)    MD5SUM="md5 -r";;
    *)          MD5SUM=md5sum
esac
if [ `$MD5SUM data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 | awk '{print $1}'` == "01edf64cd120ecc03a2745352779514c" ]; then
    echo "Downloaded the processed Wikipedia dump from the HotpotQA website. Everything's looking good, so let's extract it!"
else
    echo "The md5 doesn't seem to match what we expected, try again?"
    exit 1
fi
cd data
tar -xjvf enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
# clean up
rm enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
echo 'Done!'
