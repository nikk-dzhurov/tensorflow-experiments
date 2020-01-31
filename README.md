# Experimental tensorflow projects

## Start TF container
`./tools/start_tf_container.sh`

## Dataset preparations:
`./tools/exec_model.sh --mode="prepare_data"`

## Start training
`./tools/exec_model.sh --mode="train_eval" --clean=1 --steps 1200 --epochs=10`
