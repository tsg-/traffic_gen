#!/bin/bash

#for i in 1 2 4 8 16 32 ; do
for i in 1 ; do
    echo ./traffic_gen.py --threads $i --connections 1 --host 192.168.0.55 --port 8080 --duration 300 --ramp 30 --test-type duration --random --max-rand-obj 750 --output-dir . --req-dist gauss | tee ./traffic_gen.log
done
