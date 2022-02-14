#!/usr/bin/env bash

# If you ran the script on the vm
export PYTHONPATH=/home/romaks/pop_vm/

# If locally
export PYTHONPATH=/home/romaks/Studie/Thesis/pop/

date=$(date +%Y%m%d_%H_%M)

##############
#     MLP    #
##############

echo "starting training"
python3 experiments/run_TD.py --save-results-path "./storage/results/"$date"/run_0/"  --use-sgd False --use-ls1 False --sequence-length 1 --use-double-dqn True
echo "ending training"

#echo "starting training"
#python3 experiments/run_battery.py --save-results-path "./storage/results/"$date"/run_1/"  --use-sgd True --use-ls1 False --sequence-length 1
#echo "ending training"

#echo "starting training"
#python3 experiments/run_battery.py --save-results-path "./storage/results/"$date"/run_2/"  --use-sgd False --use-ls1 True --sequence-length 30
#echo "ending training"
#
#echo "starting training"
#python3 experiments/run_battery.py --save-results-path"./storage/results/"$date"/run_3/" --use-sgd True --use-ls1 True --sequence-length 30
#echo "ending training"


##############
#    LSTM    #
##############

echo "starting training"
python3 experiments/run_TD.py --save-results-path "./storage/results/"$date"/run_0_rec/"  --use-sgd False --use-ls1 False --sequence-length 30 --use-double-dqn True
echo "ending training"

#echo "starting training"
#python3 experiments/run_battery.py --save-results-path "./storage/results/"$date"/run_1_rec/"  --use-sgd True --use-ls1 False --sequence-length 30
#echo "ending training"

