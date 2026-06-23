set jour="2025-08-09" 
set time_index=12
set event="heat"
python plot_maps.py --date %jour% --time-index %time_index% --model "Interpolation"
python plot_maps.py --date %jour% --time-index %time_index% --model "PyESD"
python plot_maps.py --date %jour% --time-index %time_index% --model "DeepSD"
python plot_maps.py --date %jour% --time-index %time_index% --model "ESRGAN"
python plot_maps.py --date %jour% --time-index %time_index% --model "CorrDiff"
python plot_maps.py --date %jour% --time-index %time_index%

python analyse_extreme.py --date %jour% --time-index %time_index% --type_event %event% --model "Interpolation"
python analyse_extreme.py --date %jour% --time-index %time_index% --type_event %event% --model "PyESD"
python analyse_extreme.py --date %jour% --time-index %time_index% --type_event %event% --model "DeepSD"
python analyse_extreme.py --date %jour% --time-index %time_index% --type_event %event% --model "ESRGAN"
python analyse_extreme.py --date %jour% --time-index %time_index% --type_event %event% --model "CorrDiff"
python analyse_extreme.py --date %jour% --time-index %time_index% --type_event %event% 