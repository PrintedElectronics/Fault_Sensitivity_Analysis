eval "$(conda shell.bash hook)"
conda info --envs
conda activate /home/flo/anaconda3/envs/Qkeras

dataset=$1
design_folder=$2

start=`date +%s`
python3 stack-at-fault.py $dataset single $design_folder
python3 stack-at-fault.py $dataset double $design_folder
python3 stack-at-fault.py $dataset quatre $design_folder
end=`date +%s`

echo Execution time was `expr $end - $start` seconds.