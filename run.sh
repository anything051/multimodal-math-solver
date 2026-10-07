#/root/miniconda3/bin/python
echo "Start to run the code..."

${conda_path}/envs/pytorch-env/bin/python run.py \
  $IMAGE_INPUT_DIR \
  $QUERY_PATH \
  $OUPUT_PATH