#!/bin/bash

./tools/exec_model.sh --mode="train_eval" --clean=1

sed -i 's/0.3/0.5/g' src/stl10.py

./tools/exec_model.sh --mode="train_eval" --clean=1

sed -i 's/0.5/0.7/g' src/stl10.py

./tools/exec_model.sh --mode="train_eval" --clean=1

poweroff

