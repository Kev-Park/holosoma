# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Use CONDA_ENV_NAME if provided, otherwise default to "hssim"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsretargeting}
echo "conda environment name is set to: $CONDA_ENV_NAME"

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ENVS_DIR/$CONDA_ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_retargeting_$CONDA_ENV_NAME
echo "SENTINEL_FILE: $SENTINEL_FILE"

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Create the conda environment in CONDA_ENVS_DIR
  mkdir -p $CONDA_ENVS_DIR
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda create -y -p $ENV_ROOT python=3.11 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate $ENV_ROOT

  # Install holosoma_retargeting
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma_retargeting
  touch $SENTINEL_FILE
fi
