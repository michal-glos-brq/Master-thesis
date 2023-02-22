# X11 shared volume for displayting content
X11 := /tmp/.X11-unix:/tmp/.X11-unix:ro
# Directory with source files and notebook. mount argument for mounting it
SHARE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/src
MOUNT := src=${SHARE_DIR},target=/xglosm01/shared,type=bind

build:
	docker build . -t gfootball

start: allow_display
# Start the docker image, allow gpus (nvidia-containers) forward Xserver bindings  /home/mglos/skola/Diplomka/dip
	docker run --gpus all -e DISPLAY=${DISPLAY} -it -v ${X11} --mount ${MOUNT} gfootball bash

allow_display:
# Allow applications "not on this machine" (read: in a container) to join local Xserver
	xhost +"local:docker@"

play: allow_display
	docker run --gpus all -e DISPLAY=${DISPLAY} -it -v ${X11}  gfootball python3 -m gfootball.play_game --action_set=full
