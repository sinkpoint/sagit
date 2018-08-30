#!/bin/sh:
IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost $IP
docker run -e DISPLAY=$IP:0 --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -i -t sagit /bin/bash
# docker run --net=host -e DISPLAY=$IP:0 --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -i -t sagit /bin/bash