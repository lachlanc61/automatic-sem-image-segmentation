#!/bin/bash

IMAGENAME='sem/tf2.9:0.1'

docker build . -t "$IMAGENAME"
