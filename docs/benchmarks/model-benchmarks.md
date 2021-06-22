---
id: model-benchmarks
---

# Model Benchmarks

## Benchmarking list

### GPT-2 models

### BERT models

### LSTM models

### CNN models


## Benchmarking metrics

<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
      </td>
      <td>
        <b>Micro Benchmark</b>
        <img src={require('../assets/bar.png').default}/>
      </td>
      <td>
        <b>Model Benchmark</b>
        <img src={require('../assets/bar.png').default}/>
      </td>
    </tr>
    <tr valign="top">
      <td align="center" valign="middle">
        <b>Metrics</b>
      </td>
      <td>
        <ul><li><b>Computation Benchmark</b></li>
          <ul><li><b>Kernel Performance</b></li>
            <ul>
              <li>GFLOPS</li>
              <li>TensorCore</li>
              <li>cuBLAS</li>
              <li>cuDNN</li>
            </ul>
          </ul>
          <ul><li><b>Kernel Launch Time</b></li>
            <ul>
              <li>Kernel_Launch_Event_Time</li>
              <li>Kernel_Launch_Wall_Time</li>
            </ul>
          </ul>
          <ul><li><b>Operator Performance</b></li>
            <ul><li>MatMul</li><li>Sharding_MatMul</li></ul>
          </ul>
          <ul><li><b>Memory</b></li>
            <ul><li>H2D_Mem_BW_&lt;GPU ID&gt;</li>
              <li>H2D_Mem_BW_&lt;GPU ID&gt;</li></ul>
          </ul>
        </ul>
        <ul><li><b>Communication Benchmark</b></li>
          <ul><li><b>Device P2P Bandwidth</b></li>
            <ul><li>P2P_BW_Max</li><li>P2P_BW_Min</li><li>P2P_BW_Avg</li></ul>
          </ul>
          <ul><li><b>RDMA</b></li>
            <ul><li>RDMA_Peak</li><li>RDMA_Avg</li></ul>
          </ul>
          <ul><li><b>NCCL</b></li>
            <ul><li>NCCL_AllReduce</li></ul>
            <ul><li>NCCL_AllGather</li></ul>
            <ul><li>NCCL_broadcast</li></ul>
            <ul><li>NCCL_reduce</li></ul>
            <ul><li>NCCL_reduce_scatter</li></ul>
          </ul>
        </ul>
        <ul><li><b>Computation-Communication Benchmark</b></li>
          <ul><li><b>Mul_During_NCCL</b></li><li><b>MatMul_During_NCCL</b></li></ul>
        </ul>
        <ul><li><b>Storage Benchmark</b></li>
          <ul><li><b>Disk</b></li>
            <ul>
              <li>Read/Write</li><li>Rand_Read/Rand_Write</li>
              <li>R/W_Read</li><li>R/W_Write</li><li>Rand_R/W_Read</li><li>Rand_R/W_Write</li>
            </ul>
          </ul>
        </ul>
      </td>
      <td>
        <ul><li><b>CNN models</b></li>
          <ul>
            <li><b>ResNet</b></li>
              <ul><li>ResNet-50</li><li>ResNet-101</li><li>ResNet-152</li></ul>
          </ul>
          <ul>
            <li><b>DenseNet</b></li>
              <ul><li>DenseNet-169</li><li>DenseNet-201</li></ul>
          </ul>
          <ul>
            <li><b>VGG</b></li>
              <ul><li>VGG-11</li><li>VGG-13</li><li>VGG-16</li><li>VGG-19</li></ul>
          </ul>
          <ul><li><b>Other CNN models</b></li><ul><li>...</li></ul></ul>
        </ul>
        <ul><li><b>BERT models</b></li>
          <ul><li><b>BERT</b></li><li><b>BERT_LARGE</b></li></ul>
        </ul>
        <ul><li><b>LSTM</b></li></ul>
        <ul><li><b>GPT-2</b></li></ul>
      </td>
    </tr>
  </tbody>
</table>
