#!/bin/bash

IMAGENAME='sem/tf2.9:0.1'
BIND_DIR='./trial_data/Eline_BT386C2/'
CMD=$1

echo $CMD

docker run \
	-it --rm --gpus=all \
     -e LOCAL_UID=$(id -u $USER) \
     -e LOCAL_GID=$(id -g $USER) \
	--mount type=bind,source="$(pwd)"/"$BIND_DIR",target=/sourcedata \
	"$IMAGENAME" \
	"$CMD"

#--mount "$BIND_DIR" "$CMD"
