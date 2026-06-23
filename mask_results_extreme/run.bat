@echo off
pushd "%~dp0"
python -B analyse_extreme_batch.py --type-event all
popd
