#!/bin/bash
nohup python $1 &>> "logs/$(basename $1).log" &
