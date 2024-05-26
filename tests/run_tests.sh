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
$PYCMD test_datasets_gcs.py $@
$PYCMD test_datasets_utils.py $@
$PYCMD test_datasets_infer.py $@

echo "Testing loaders"
$PYCMD test_loaders.py $@

echo "Testing models"
$PYCMD test_models.py $@

echo "Testing readers"
$PYCMD test_readers.py $@

echo "Testing samplers"
$PYCMD test_samplers.py $@
$PYCMD test_samplers_slices.py $@
$PYCMD test_samplers_patch.py $@

echo "Testing trainers"
$PYCMD test_trainers.py $@
$PYCMD test_trainers_torch.py $@

echo "Testing transforms"
$PYCMD test_transforms.py $@
$PYCMD test_transforms_random.py $@

echo "Testing explainers"
$PYCMD test_explainers.py $@

echo "Testing workflows"
$PYCMD test_workflows.py $@

echo "Testing bugs"
$PYCMD test_bugs.py $@

if [[ $COVERAGE -eq 1 ]]; then
    coverage combine
    coverage xml
fi


popd
