---
id: system-config
---

# System Config Info

This tool is to collect the system information automatically on the tested GPU nodes including the following hardware categories:

- [System](#system)
- [Memory](#memory)
- [CPU](#cpu)
- [Disk](#disk)
- [Networking](#networking)
- [Accelerator](#accelerator)
- [PCIe](#pcie)

## Usage

### Usage on local machine

1. [Install SuperBench](../getting-started/installation.mdx) on the local machine using root privilege.

2. Start to collect the sys info using `sb node info --output-dir ${output-dir}` command using root privilege.

3. After the command finished, you can find the output system info json file `sys-info.json` of local node under \${output_dir}.

### Usage on multiple remote machines

1. [Install SuperBench](../getting-started/installation.mdx) on the local machine.

2. [Deploy SuperBench](../getting-started/run-superbench.md#deploy) onto the remote machines.

2. Prepare the host file of the tested GPU nodes using [Ansible Inventory](../getting-started/configuration.md#ansible-inventory) on the local machine.

3. After installing the Superbnech and the host file is ready, you can start to collect the sys info automatically using  `sb run --get-info` command. The detailed command can be found from [SuperBench CLI](../cli.md).

  ```
  sb run --get-info -f host.ini --output-dir ${output-dir} -C superbench.enable=none
  ```

4. After the command finished, you can find the output system info json file `sys-info.json` of each node under \${output_dir}/nodes/${node_name}.

## Parameter and Details

### System

<table>
  <tbody>
    <tr align="centor" valign="bottom">
      <td>
        <b>SubCategory</b>
      </td>
      <td>
        <b>Key</b>
      </td>
      <td>
        <b>Command</b>
      </td>
      <td>
        <b>Description</b>
      </td>
      <td>
        <b>Example</b>
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="4">
        <b>OS</b>
      </td>
      <td>system-manufacturer</td>
      <td>dmidecode -s system-manufacturer</td>
      <td>manufacturer of the system</td>
      <td>Microsoft Corporation</td>
    </tr>
      <td>system-product name(virtual machine)</td>
      <td>dmidecode -s system-product-name</td>
      <td>product name or virtual machine</td>
      <td>Virtual Machine</td>
    <tr>
      <td>operating_system</td>
      <td>cat /proc/version</td>
      <td>version of current running os</td>
      <td>Ubuntu 9.3.0-17ubuntu1~20.04</td>
    </tr>
    <tr>
      <td>uname</td>
      <td>uname</td>
      <td>short for system information</td>
      <td>Linux sb-test-wu-000000 5.8.0-1039-azure #42~20.04.1-Ubuntu</td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="2">
        <b>Docker</b>
      </td>
      <td>docker_server_version</td>
      <td>docker version</td>
      <td>server version of docker engine</td>
      <td>20.10.3</td>
    </tr>
    <tr>
      <td>docker_client_version</td>
      <td>docker version</td>
      <td>client version of docker engine</td>
      <td>20.10.3</td>
    </tr>
    <tr>
      <td align="center"><b>VM</b></td>
      <td>vmbus</td>
      <td>lsvmbus</td>
      <td>devices attached to the Hyper-V VMBus</td>
      <td>
        "VMBUS ID  1": "[Dynamic Memory]",<br />
        "VMBUS ID  2": "Synthetic mouse",<br />
        ...
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="2">
        <b>Kernel</b>
      </td>
      <td>kernel_modules</td>
      <td>lsmod</td>
      <td>list of active kernel modules</td>
      <td>
        "Module": "binfmt_misc",<br />
        "Size": "24576",<br />
        "Used": "1"<br />
        ...
      </td>
    </tr>
    <tr>
      <td>kernel_parameters</td>
      <td>sysctl</td>
      <td>kernel parameters</td>
      <td>
        "abi.vsyscall32": "1",<br />
        "debug.exception-trace": "1",<br />
        ...
      </td>
    </tr>
    <tr>
      <td align="center"><b>DMI</b></td>
      <td>dmidecode</td>
      <td>dmidecode</td>
      <td>DMI table dump (info on hardware components)</td>
      <td>"dmidecode": "# dmidecode 3.2\nGetting SMBIOS data from sysfs..."</td>
    </tr>
  </tbody>
</table>

### Memory

<table>
  <tbody>
    <tr align="centor" valign="bottom">
      <td>
        <b>SubCategory</b>
      </td>
      <td>
        <b>Key</b>
      </td>
      <td>
        <b>Command</b>
      </td>
      <td>
        <b>Description</b>
      </td>
      <td>
        <b>Example</b>
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="6">
        <b>General</b>
      </td>
      <td>model</td>
      <td>dmidecode -t memory</td>
      <td>distinct model name of the memory</td>
      <td>Samsung M393A4K40DB3-CWE</td>
    </tr>
    <tr>
      <td>type</td>
      <td>dmidecode -t memory</td>
      <td>distinct type of memory</td>
      <td>DDR4-3200</td>
    </tr>
    <tr>
      <td>clock frequency</td>
      <td>dmidecode -t memory</td>
      <td>distinct clock frequency of memory</td>
      <td>3200 MT/s</td>
    </tr>
    <tr>
      <td>channels</td>
      <td>dmidecode -t memory</td>
      <td>the number of memory chips</td>
      <td>16</td>
    </tr>
    <tr>
      <td>capacity</td>
      <td>lsmem</td>
      <td>the total capacity of memory</td>
      <td>511.9G</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>lsmem</td>
      <td>the block size of memory</td>
      <td>128M</td>
    </tr>
  </tbody>
</table>

### CPU

<table>
  <tbody>
    <tr align="centor" valign="bottom">
      <td>
        <b>SubCategory</b>
      </td>
      <td>
        <b>Key</b>
      </td>
      <td>
        <b>Command</b>
      </td>
      <td>
        <b>Description</b>
      </td>
      <td>
        <b>Example</b>
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="16">
        <b>General</b>
      </td>
      <td>archeticture</td>
      <td>lscpu</td>
      <td>architecture of cpu</td>
      <td>x86_64</td>
    </tr>
    <tr>
      <td>model name</td>
      <td>lscpu</td>
      <td>model name of cpu</td>
      <td>AMD EPYC 7662 64-Core Processor</td>
    </tr>
    <tr>
      <td>cpu op-mode</td>
      <td>lscpu</td>
      <td>cpu mode: 32bit/64bit</td>
      <td>32-bit, 64-bit</td>
    </tr>
    <tr>
      <td>byte order</td>
      <td>lscpu</td>
      <td>byte order</td>
      <td>Little Endian</td>
    </tr>
    <tr>
      <td>address size</td>
      <td>lscpu</td>
      <td>size of address</td>
      <td>48 bits physical, 48 bits virtual</td>
    </tr>
    <tr>
      <td>cpus</td>
      <td>lscpu</td>
      <td>logical cpu cores count</td>
      <td>256</td>
    </tr>
    <tr>
      <td>On-line CPU(s) list</td>
      <td>lscpu</td>
      <td>on-line logical cpu cores</td>
      <td>0-255</td>
    </tr>
    <tr>
      <td>Thread(s) per core</td>
      <td>lscpu</td>
      <td>thread per core</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Core(s) per socket</td>
      <td>lscpu</td>
      <td>core per socket</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Socket(s)</td>
      <td>lscpu</td>
      <td>socket count</td>
      <td>2</td>
    </tr>
    <tr>
      <td>NUMA node(s)</td>
      <td>lscpu</td>
      <td>numa node count</td>
      <td>4</td>
    </tr>
    <tr>
      <td>L&ltx&gt caches</td>
      <td>lscpu</td>
      <td>cache size</td>
      <td>"L1d cache": "4 MiB",
        "L1i cache": "4 MiB",
        "L2 cache": "64 MiB",
        "L3 cache": "512 MiB"</td>
    </tr>
    <tr>
      <td>NUMA node&ltx&gt CPU(s)</td>
      <td>lscpu</td>
      <td>cpu core list of the numa node</td>
      <td>"NUMA node0 CPU(s)": "0-31,128-159",
        "NUMA node1 CPU(s)": "32-63,160-191",
        "NUMA node2 CPU(s)": "64-95,192-223",
        "NUMA node3 CPU(s)": "96-127,224-255"</td>
    </tr>
    <tr>
      <td>Flags</td>
      <td>lscpu</td>
      <td>cpu flags</td>
      <td> fpu vme de pse tsc msr pae mce cx8 apic ...</td>
    </tr>
    <tr>
      <td>max_speed</td>
      <td>sudo dmidecode -t processor | grep "Speed"</td>
      <td>distinct cpu max frequency</td>
      <td>3700 MHz</td>
    </tr>
    <tr>
      <td>current_speed</td>
      <td>sudo dmidecode -t processor | grep "Speed"</td>
      <td>distinct cpu current frequency</td>
      <td>2000 MHz</td>
    </tr>
  </tbody>
</table>

### Disk

<table>
  <tbody>
    <tr align="centor" valign="bottom">
      <td>
        <b>SubCategory</b>
      </td>
      <td>
        <b>Key</b>
      </td>
      <td>
        <b>Command</b>
      </td>
      <td>
        <b>Description</b>
      </td>
      <td>
        <b>Example</b>
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="6">
        <b>FileSystem</b>
      </td>
      <td>filesystem</td>
      <td>df -Th</td>
      <td>the name/path of the filesystem</td>
      <td>/dev/nvme0n1p2</td>
    </tr>
    <tr>
      <td>avail</td>
      <td>df -Th</td>
      <td>avail size of the filesystem</td>
      <td>1.4T</td>
    </tr>
    <tr>
      <td>size</td>
      <td>df -Th</td>
      <td>total size of the filesystem</td>
      <td>1.8T</td>
    </tr>
    <tr>
      <td>type</td>
      <td>df -Th</td>
      <td>the type of the filesystem</td>
      <td>ext4</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>blockdev --getbsz /dev/&ltdevice&gt</td>
      <td>the block size of the filesytem</td>
      <td>4096</td>
    </tr>
    <tr>
      <td>4k_alignment</td>
      <td>4kDEVICE=/dev/sdb1 do parted $DEVICE align-check opt 1; done_alignment</td>
      <td>whether the file system is 4k alignment</td>
      <td>1 aligned</td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="5">
        <b>BlockDevice</b>
      </td>
      <td>name</td>
      <td>lsblk -e 7 -o NAME,ROTA,SIZE,MODEL </td>
      <td>the name of the block device</td>
      <td>nvme0n1</td>
    </tr>
    <tr>
      <td>model</td>
      <td>lsblk -e 7 -o NAME,ROTA,SIZE,MODEL </td>
      <td>the model name of the block device</td>
      <td>VO001920KXAVP</td>
    </tr>
    <tr>
      <td>rotational</td>
      <td>lsblk -e 7 -o NAME,ROTA,SIZE,MODEL </td>
      <td>whether rotational, thai is HDD or SSD</td>
      <td>0</td>
    </tr>
    <tr>
      <td>size</td>
      <td>lsblk -e 7 -o NAME,ROTA,SIZE,MODEL </td>
      <td>the total size of the block device</td>
      <td>1.8T</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>fdisk -l -u /dev/{} | grep "Sector size"</td>
      <td>the sector size of the block device</td>
      <td>Sector size (logical/physical): 512 bytes / 512 bytes</td>
    </tr>
    <tr>
      <td align="center"><b>General</b></td>
      <td>mapping</td>
      <td>mount</td>
      <td>mount relationship between filesystem and block device</td>
      <td></td>
    </tr>
  </tbody>
</table>

### Networking

<table>
  <tbody>
    <tr align="centor" valign="bottom">
      <td>
        <b>SubCategory</b>
      </td>
      <td>
        <b>Key</b>
      </td>
      <td>
        <b>Command</b>
      </td>
      <td>
        <b>Description</b>
      </td>
      <td>
        <b>Example</b>
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="6">
        <b>NIC</b>
      </td>
      <td>nic_logical_name</td>
      <td>lshw -c network</td>
      <td>logical name of the nic</td>
      <td>ib1</td>
    </tr>
    <tr>
      <td>nic_model</td>
      <td>lshw -c network</td>
      <td>model name of the nic</td>
      <td>Mellanox Technologies MT28908 Family [ConnectX-6]</td>
    </tr>
    <tr>
      <td>nic_firmware</td>
      <td>lshw -c network</td>
      <td>fw version</td>
      <td>20.30.1004 (MT_0000000594)</td>
    </tr>
    <tr>
      <td>nic_driver</td>
      <td>lshw -c network</td>
      <td>driver version</td>
      <td>mlx5_core[ib_ipoib] 5.3-1.0.0</td>
    </tr>
    <tr>
      <td>nic_speed</td>
      <td>lshw -c network</td>
      <td>speed spec of the nic</td>
      <td>200 Gbit/s</td>
    </tr>
    <tr>
      <td>nic_disabled</td>
      <td>lshw -c network</td>
      <td>whether diabled</td>
      <td>false</td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="2">
        <b>IB</b>
      </td>
      <td>device_info</td>
      <td>ibv_devinfo -v</td>
      <td>list of device information for each ib device</td>
      <td>"hca_id:\tmlx5_0": ...</td>
    </tr>
    <tr>
      <td>device_status</td>
      <td>ibstat</td>
      <td>list of device status for each ib device</td>
      <td>
        "CA 'mlx5_0'": ...</td>
    </tr>
    <tr>
      <td align="center"><b>General</b></td>
      <td>ofed_version</td>
      <td>ofed_info  -s</td>
      <td>the version of ofed</td>
      <td>MLNX_OFED_LINUX-5.3-1.0.5.0:</td>
    </tr>
  </tbody>
</table>

### Accelerator

<table>
  <tbody>
    <tr align="centor" valign="bottom">
      <td>
        <b>SubCategory</b>
      </td>
      <td>
        <b>Key</b>
      </td>
      <td>
        <b>Command</b>
      </td>
      <td>
        <b>Description</b>
      </td>
      <td>
        <b>Example(NVIDIA)</b>
      </td>
      <td>
        <b>Example(AMD)</b>
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="5">
        <b>General</b>
      </td>
      <td>driver_version</td>
      <td>nvidia-smi -q -x/rocm-smi -a</td>
      <td>driver version</td>
      <td>460.27.04</td>
      <td>5.9.25</td>
    </tr>
    <tr>
      <td>topology</td>
      <td>nvidia-smi topo -m/rocm-smi --showtopo</td>
      <td>gpu connection topology (nvidia only)</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td>nvidia-container-runtime_version</td>
      <td>nvidia-container-runtime -v</td>
      <td>version of nvidia-container-runtime (nvidia only)</td>
      <td>1.0.0-rc92</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>nvidia-fabricmanager_version</td>
      <td>nv-fabricmanager --version</td>
      <td>version of nvidia-fabricmanager (nvidia only)</td>
      <td>460.27.04</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>nv_peer_mem_version</td>
      <td>dpkg -l | grep 'nvidia-peer-memory'</td>
      <td>version of nv_peer_mem (nvidia only)</td>
      <td>1.1-0</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="2">
        <b>GPUCard</b>
      </td>
      <td>rocm_info</td>
      <td>rocm-smi -a & rocm-smi --showmeminfo vram</td>
      <td>amd gpu info of each gpu&lsindex&gt, including firmware, frequency, memory, etc. (amd only)</td>
      <td>N/A</td>
      <td>"card0": ...<br />"card1": ...</td>
    </tr>
    <tr>
      <td>nvidia_info</td>
      <td>nvidia-smi -q</td>
      <td>nvidia gpu info list of each gpu, including firmware, frequency, memory, etc. (nvidia only)</td>
      <td>
        "timestamp": "Fri Aug 20 05:36:24 2021",<br />
        "driver_version": "460.27.04",<br />
        "cuda_version": "11.2",<br />
        "attached_gpus": "8",<br />
        "gpu": [...]<br />
        ...<br />
      </td>
      <td>N/A</td>
    </tr>
  </tbody>
</table>

### PCIe

<table>
  <tbody>
    <tr align="centor" valign="bottom">
      <td>
        <b>SubCategory</b>
      </td>
      <td>
        <b>Key</b>
      </td>
      <td>
        <b>Command</b>
      </td>
      <td>
        <b>Description</b>
      </td>
      <td>
        <b>Example</b>
      </td>
    </tr>
    <tr>
      <td align="center" valign="middle" rowspan="2">
        <b>General</b>
      </td>
      <td>topology</td>
      <td>lspci -t -vvv</td>
      <td>topology of installed PCI devices</td>
      <td>/</td>
    </tr>
    <tr>
      <td>device_info</td>
      <td>lspci -vvv</td>
      <td>device info on installed PCI devices</td>
      <td>00:00.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex...</td>
    </tr>
  </tbody>
</table>
