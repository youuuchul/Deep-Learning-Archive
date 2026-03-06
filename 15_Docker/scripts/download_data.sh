#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

curl -L "https://bakey-api.codeit.kr/api/files/resource?root=static&seqId=14111&version=1&directory=/mission15_train.csv&name=mission15_train.csv" -o data/raw/mission15_train.csv
curl -L "https://bakey-api.codeit.kr/api/files/resource?root=static&seqId=14111&version=1&directory=/mission15_test.csv&name=mission15_test.csv" -o data/raw/mission15_test.csv

echo "download complete: data/raw/mission15_train.csv, data/raw/mission15_test.csv"
