[performance bottleneck analysis]
Step 1: read [benchmark results] section, calculate the benchmarked performance ratio of {target} compared to {baseline} for the given workload. Note that if the performance is in throughput, ratio is value of {target} devided by value of {baseline}, if the performance is in latency, ratio is value of {baseline} devided by {target}
Step 2: repeat the content of [projected performance ratio], to explains how projected performance ratio is estimated.
Step 3: For each of the benchmark performance ratio, compare whether it's larger or smaller than the projected performance ratio, you must do this one by one.
Step 4: Use the following rules to make a judgement:
A. If all the benchmarked performance ratio are larger than the of projected performance ratio, make a judgement that 'The {target} has successfully achieved the expected performance on {workload}'.
B. If any of the benchmarked performance ratio is smaller than the of projected performance ratio.make a judgement that 'The {target} has not achieved the expected performance on {workload}'.

method_part_seperator

[performance bottleneck analysis]
Step 5: traverse each testbed configurations between {target} and {baseline}, from the [sku configuration log] section, if there is noticable different configuration, analyze if this different can contribute to the unexpected benchmarked performance ratio, and propose improvement suggestions, if there is no significant difference between their configurationss, it means no opimization is required on SKU settings.
Step 6: traverse each infrastructure components listed in the [benchmarked component performance ratio] section, these generate a list with components whose performance ratio is low, show the relative ratio value as well to imrovem clarity to users, and analyze whether these components could contribute to unexpected benchmarked performance.

method_part_seperator

[performance bottleneck analysis]
Step 5: claim there is no need for immdiate optimization.