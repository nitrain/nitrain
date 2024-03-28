#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}
COVERAGE=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -p|--python) PYCMD=$2; shift 2 ;;
        -c|--coverage) COVERAGE=1; shift 1;;
        --) shift; break ;;
        *) echo "Invalid argument: $1!" ; exit 1 ;;
    esac
done

if [[ $COVERAGE -eq 1 ]]; then
    coverage erase
    PYCMD="coverage run --parallel-mode --source nitrain "
    echo "coverage flag found. Setting command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"

echo "Testing datasets"
$PYCMD test_datasets.py $@

echo "Testing explainers"
$PYCMD test_explainers.py $@

echo "Testing loaders"
$PYCMD test_loaders.py $@

echo "Testing models"
$PYCMD test_models.py $@

echo "Testing samplers"
$PYCMD test_samplers.py $@

echo "Testing trainers"
$PYCMD test_trainers.py $@

echo "Testing transforms"
$PYCMD test_transforms.py $@


if [[ $COVERAGE -eq 1 ]]; then
    coverage combine
    coverage xml
fi


popd
