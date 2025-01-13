#!/bin/bash

function start_server(){
    echo "Starting server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_SINGLE python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_SINGLE \
    --trust-remote-code \
    --tensor-parallel-size $TENSOR_SIZE_SINGLE \
    --seed 42 \
    --port $PORT_SINGLE > ${SERVER_LOG_FILE} 2>&1 &


    # sh ${RUN_SERVER} ${CUDA_DEVICES_SINGLE} ${MODEL_PATH_SINGLE} ${TENSOR_SIZE_SINGLE} ${PORT_SINGLE} > ${SERVER_LOG_FILE} 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    echo "Waiting for server to start..."
    while ! grep "Uvicorn running on" ${SERVER_LOG_FILE}; do
        sleep 10
    done
}

function start_run_script_single() {
    echo "Starting the run script..."
    python inference.py \
    --generation_model_name ${MODEL_PATH_SINGLE} \
    --generation_model_port ${PORT_SINGLE} \
    --prompt_key ${PROMPT} \
    --use_feedback ${USE_FB} \
    --save_dir ${SAVE_DIR}
    # \
    # > "${RUN_SCRIPT_LOG_FILE}" &

    RUN_SCRIPT_PID=$!
    # echo "Run script PID: ${RUN_SCRIPT_PID}"

    # echo "Waiting for run script to complete..."
    # while ! grep "Done!" ${RUN_SCRIPT_LOG_FILE}; do
    #     sleep 10
    # done
}

# ----------------- Functions for the two-server version -----------------
# function start_two_servers() {
#     # Feedback Server
#     echo "Starting feedback server..."
#     sh ${RUN_SERVER} ${CUDA_DEVICES_FEEDBACK} ${MODEL_PATH_FEEDBACK} ${TENSOR_SIZE_TWO} ${PORT_FEEDBACK} > ${FEEDBACK_SERVER_LOG_FILE} 2>&1 &
#     FEEDBACK_SERVER_PID=$!
#     echo "Feedback server PID: ${FEEDBACK_SERVER_PID}"

#     # Generate Server
#     echo "Starting generate server..."
#     sh ${RUN_SERVER} ${CUDA_DEVICES_GENERATE} ${MODEL_PATH_GENERATE} ${TENSOR_SIZE_TWO} ${PORT_GENERATE} > ${GENERATE_SERVER_LOG_FILE} 2>&1 &
#     GENERATE_SERVER_PID=$!
#     echo "Generate server PID: ${GENERATE_SERVER_PID}"

#     # Wait for Feedback Server
#     echo "Waiting for feedback server to start..."
#     while ! grep "Uvicorn running on" ${FEEDBACK_SERVER_LOG_FILE}; do
#         sleep 10
#     done

#     # Wait for Generate Server
#     echo "Waiting for generate server to start..."
#     while ! grep "Uvicorn running on" ${GENERATE_SERVER_LOG_FILE}; do
#         sleep 10
#     done

#     echo "Both servers started successfully!"
# }

function start_two_servers() {
    # Feedback Server
    touch ${FEEDBACK_SERVER_LOG_FILE}
    touch ${GENERATE_SERVER_LOG_FILE}
    echo "Starting feedback server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_FEEDBACK python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_FEEDBACK \
    --tensor-parallel-size $TENSOR_SIZE_TWO \
    --seed 42 \
    --port $PORT_FEEDBACK > ${FEEDBACK_SERVER_LOG_FILE} 2>&1 &
    FEEDBACK_SERVER_PID=$!
    echo "Feedback server PID: ${FEEDBACK_SERVER_PID}"

    # Generate Server
    echo "Starting generate server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_GENERATE python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_GENERATE \
    --tensor-parallel-size $TENSOR_SIZE_TWO \
    --seed 42 \
    --port $PORT_GENERATE > ${GENERATE_SERVER_LOG_FILE} 2>&1 &
    GENERATE_SERVER_PID=$!
    echo "Generate server PID: ${GENERATE_SERVER_PID}"

    # Wait for Feedback Server
    echo "Waiting for feedback server to start..."
    while ! grep "Uvicorn running on" ${FEEDBACK_SERVER_LOG_FILE}; do
        sleep 10
    done

    # Wait for Generate Server
    echo "Waiting for generate server to start..."
    while ! grep "Uvicorn running on" ${GENERATE_SERVER_LOG_FILE}; do
        sleep 10
    done

    echo "Both servers started successfully!"
}

function start_run_script_two() {
    echo "Starting the run script for both servers..."
    python inference.py \
    --feedback_model_name ${MODEL_PATH_FEEDBACK} \
    --feedback_model_port ${PORT_FEEDBACK} \
    --generation_model_name ${MODEL_PATH_GENERATE} \
    --generation_model_port ${PORT_GENERATE} \
    --save_dir ${SAVE_DIR}
    # sh ${RUN_SCRIPT} ${MODEL_PATH_FEEDBACK} ${PORT_FEEDBACK} ${MODEL_PATH_GENERATE} ${PORT_GENERATE} ${PROMPT} ${SAVE_LOCATION} ${USE_FB} > ${RUN_SCRIPT_LOG_FILE} 2>&1 &
    RUN_SCRIPT_PID=$!
    # echo "Run script PID: ${RUN_SCRIPT_PID}"

    # echo "Waiting for run script to complete..."
    # while ! grep "Done!" ${RUN_SCRIPT_LOG_FILE}; do
    #     sleep 10
    # done
}

function handle_sigint() {
    echo -e "\nYou've stopped the main script. The run script process will be terminated."
    kill ${RUN_SCRIPT_PID} || true
    # Ask user if they want to kill the server
    if [ "$USE_TWO_SERVERS" == "yes" ]; then
        kill ${FEEDBACK_SERVER_PID}
        kill ${GENERATE_SERVER_PID}
        echo "Server process terminated."
    else
        kill ${SERVER_PID}
        echo "Server process terminated."
    fi
    exit 0
}


# ----------------- Main Execution -----------------
USE_TWO_SERVERS=$1
RUN_SERVER="script/run_server.sh"
FIRST_MODEL_NAME=$(basename $3)
LOG_DIR="logs"
trap handle_sigint SIGINT
# RUN_SCRIPT_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_run_script.log"
if [ "$USE_TWO_SERVERS" == "yes" ]; then
    if [ "$#" -ne 11 ]; then
        echo "Error: Incorrect number of arguments for two-server version."
        exit 1
    fi
    SECOND_MODEL_NAME=$(basename $6)

    GENERATE_SERVER_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_generate_server.log"
    FEEDBACK_SERVER_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_feedback_server.log"
    CUDA_DEVICES_FEEDBACK=$2
    MODEL_PATH_FEEDBACK=$3
    PORT_FEEDBACK=$4
    CUDA_DEVICES_GENERATE=$5
    MODEL_PATH_GENERATE=$6
    PORT_GENERATE=$7
    TENSOR_SIZE_TWO=$8
    SAVE_DIR=$9
    PROMPT=${10}
    USE_FB=${11}
    echo "CUDA_DEVICES_FEEDBACK: ${CUDA_DEVICES_FEEDBACK}"
    echo "MODEL_PATH_FEEDBACK: ${MODEL_PATH_FEEDBACK}"
    echo "PORT_FEEDBACK: ${PORT_FEEDBACK}"
    echo "CUDA_DEVICES_GENERATE: ${CUDA_DEVICES_GENERATE}"
    echo "MODEL_PATH_GENERATE: ${MODEL_PATH_GENERATE}"
    echo "PORT_GENERATE: ${PORT_GENERATE}"
    echo "TENSOR_SIZE_TWO: ${TENSOR_SIZE_TWO}"
    echo "PROMPT: ${PROMPT}"
    echo "USE_FB: ${USE_FB}"

    echo "Result will be saved in: ${SAVE_DIR}"

    start_two_servers
    start_run_script_two
    kill ${FEEDBACK_SERVER_PID}
    kill ${GENERATE_SERVER_PID}
else
    if [ "$#" -ne 8 ]; then
        echo "Error: Incorrect number of arguments for single-server version."
        exit 1
    fi

    SERVER_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_server.log"
    CUDA_DEVICES_SINGLE=$2
    MODEL_PATH_SINGLE=$3
    PORT_SINGLE=$4
    TENSOR_SIZE_SINGLE=$5
    SAVE_DIR=$6
    PROMPT=${7}
    USE_FB=${8}
    echo "Result will be saved in: ${SAVE_DIR}"
    
    start_server
    start_run_script_single
    echo "Saved in: ${SAVE_DIR}"
    kill ${SERVER_PID}
fi


