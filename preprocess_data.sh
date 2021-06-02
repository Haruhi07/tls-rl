SOURCE_DATASET=Dataset/tls-rl-dataset
TARGET_DATASET=./dataset
DATASET=t1
HEIDELTIME=venv/lib/python3.8/site-packages/tilse/tools/heideltime

python preprocess_data.py --heideltime $HEIDELTIME --source $SOURCE_DATASET/$DATASET --target $TARGET_DATASET/$DATASET