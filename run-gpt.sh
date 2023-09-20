#!/bin/bash

user="ficoguti"
eval cd ~$user

source ~/workspace/sb/bin/activate

sb run -f workspace/superbenchmark/local.ini -c workspace/superbenchmark/default.yaml
