#!/bin/bash
for i in `seq 0 99`;
do
        echo "epoch $i"
        python3 testNetwork.py "epochs/epoch$i.data"
done  