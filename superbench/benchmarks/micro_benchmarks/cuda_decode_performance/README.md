# CUDA Decoding performance microbenchmark

this benchmark is revised based on AppDecPerf in NVIDIA Video Codec SDK 12.1.14

## Installation

- cd samples && mkdir build && cd build 
- cmake .. 
- cmake --build . --target AppDecPerf

## Usage

- Use single thread to decode single video file repeatedly 
```
AppDecPerf -i {path/video.mp4} -thread 1 -total 100
```

- Use multiple threads with multiple CUDA contexts to decode single video file repeatedly
```
AppDecPerf -i {path/video.mp4} -thread 10 -total 100
```

- Use multiple threads with single shared CUDA contexts to decode single video file repeatedly
```
AppDecPerf -i {path/video.mp4} -thread 10 -total 100 -single
```

- Use multiple threads with single shared CUDA CUDA contexts to decode multiple video file continuously in streaming
```
AppDecPerf -multi_input {path/video_path_list_file} -thread 10 -total 100 -single
```
