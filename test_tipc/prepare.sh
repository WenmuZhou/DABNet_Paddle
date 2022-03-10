
   
#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python_name=$(func_parser_value "${lines[2]}")

if [ ${MODE} = "lite_train_lite_infer" ];then
    # pretrain lite train data
    rm -rf DABNet_data/ DABNet_data.tar
    ${python_name} -m pip install -r requirements.txt
    wget -nc -P ./ https://paddle-model-ecology.bj.bcebos.com/whole_chain/framework_exp_2022/DABNet_data.tar --no-check-certificate
    tar -xf DABNet_data.tar
fi