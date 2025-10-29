[Task assignment]
Your task is to calculate the projected performance ratio of {target} over {baseline} for {workload}

To do this, you need to use these steps:
step 1, look into the [specification] part, look for GEMM FP16 peak performance of TFLOPs for both target = {target} and baseline = {baseline}, calculate the projected performance ratio = ({target}'s GEMM FP16 performance) divided by ({baseline}'s GEMM FP16 performance), 
step 2, generate an conclusion based on the calculated ratio, please keep only the final conclusion using the below format template, you don't need to output the steps.

[format template]
The anticipated performance ratio of {target} compared to {baseline} is projected to be xx for {workload}, this is calculated from the peak GEMM throughput of {target} over {baseline}, indicating the best relative performance we can get on {target} assuming all optimizations applied.

The followings are some examples:
[Example 1]
task is: Your task is to calculate the projected performance ratio of 4090Ti over 2080Ti for bert
output: 'The anticipated performance ratio of 4090Ti compared to 2080Ti is projected to be 50% for bert, this is calculated from the peak GEMM throughput of {target} over {baseline}, indicating the best relative performance we can get on {target} assuming all optimizations applied.'

[specification]
{specification}