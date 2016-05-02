#!/bin/bash
ffmpeg -y -i $1 -qscale 0 "still.%04d.jpg"
