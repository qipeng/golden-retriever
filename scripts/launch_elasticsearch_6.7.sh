cd elasticsearch-6.7.0
bin/elasticsearch 2>&1 >/dev/null &
while ! curl -I localhost:9200 2>/dev/null;
do
  sleep 2;
done
