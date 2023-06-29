#!/bin/bash
nodes=$(numactl --show | awk -F: '/^cpubind/ {print $2;}' | sed -e 's/^ //g' -e 's/ $//g' | tr ' ' ',')
exec numactl --interleave=${nodes} $@
