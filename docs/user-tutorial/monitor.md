---
id: monitor
---

# Monitor

SuperBench provides a `Monitor` module to collect the system metrics and detect the failure during the benchmarking. Currently this monitor supports CUDA platform only. Users can enable it in the config file.

## Configuration

```yaml
superbench:
  monitor:
    enable: bool
    sample_duration: int
    sample_interval: int
```

### `enable`

Whether enable the monitor module or not.

### `sample_duration`

Calculate the average metrics during sample_duration seconds, such as CPU usage and NIC bandwidth.

### `sample_interval`

Do sampling every sample_interval seconds.

## Metrics

Monitor module will generate the data in jsonlines format, and each line is in json format, including the following metrics:

| Name                              | Unit       | Description                                                 |
|-----------------------------------|------------|-------------------------------------------------------------|
| time                              | datetime   | The timestamp to collect the system metrics.                |
| cpu_usage                         | percentage | The average CPU utilization.                                |
| gpu_usage                         | percentage | The GPU utilization.                                        |
| gpu_temperature                   | celsius    | The GPU temperature.                                        |
| gpu_power_limit                   | watt       | The GPU power limitation.                                   |
| gpu_mem_used                      | MB         | The used GPU memory.                                        |
| gpu_mem_total                     | MB         | The total GPU memory.                                       |
| gpu_corrected_ecc                 | count      | Number of corrected (single bit) ECC error.                 |
| gpu_uncorrected_ecc               | count      | Number of uncorrected (double bit) ECC error.               |
| gpu_remap_correctable_error       | count      | Number of rows remapped due to correctable errors.          |
| gpu_remap_uncorrectable_error     | count      | Number of rows remapped due to uncorrectable.               |
| gpu_remap_max                     | count      | Number of banks with 8 available remapping resource.        |
| gpu_remap_high                    | count      | Number of banks with 7 available remapping resource.        |
| gpu_remap_partial                 | count      | Number of banks with 2~6 available remapping resource.      |
| gpu_remap_low                     | count      | Number of banks with 1 available remapping resource.        |
| gpu_remap_none                    | count      | Number of banks with 0 available remapping resource.        |
| {device}_receive_bw               | bytes/s    | Network receive bandwidth.                                  |
| {device}_transmit_bw              | bytes/s    | Network transmit bandwidth.                                 |
