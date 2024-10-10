"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[301],{3905:function(e,t,r){r.d(t,{Zo:function(){return c},kt:function(){return d}});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var p=n.createContext({}),u=function(e){var t=n.useContext(p),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},c=function(e){var t=u(e.components);return n.createElement(p.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},s=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,p=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),s=u(r),d=a,h=s["".concat(p,".").concat(d)]||s[d]||m[d]||o;return r?n.createElement(h,i(i({ref:t},c),{},{components:r})):n.createElement(h,i({ref:t},c))}));function d(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=s;var l={};for(var p in t)hasOwnProperty.call(t,p)&&(l[p]=t[p]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var u=2;u<o;u++)i[u]=r[u];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}s.displayName="MDXCreateElement"},2314:function(e,t,r){r.r(t),r.d(t,{contentTitle:function(){return p},default:function(){return s},frontMatter:function(){return l},metadata:function(){return u},toc:function(){return c}});var n=r(7462),a=r(3366),o=(r(7294),r(3905)),i=["components"],l={slug:"release-sb-v0.10",title:"Releasing SuperBench v0.10",author:"Peng Cheng",author_title:"SuperBench Team",author_url:"https://github.com/cp5555",author_image_url:"https://github.com/cp5555.png",tags:["superbench","announcement","release"]},p=void 0,u={permalink:"/superbenchmark/blog/release-sb-v0.10",editUrl:"https://github.com/microsoft/superbenchmark/edit/main/website/blog/2023-12-31-release-0-10.md",source:"@site/blog/2023-12-31-release-0-10.md",title:"Releasing SuperBench v0.10",description:"We are very happy to announce that SuperBench 0.10.0 version is officially released today!",date:"2023-12-31T00:00:00.000Z",formattedDate:"December 31, 2023",tags:[{label:"superbench",permalink:"/superbenchmark/blog/tags/superbench"},{label:"announcement",permalink:"/superbenchmark/blog/tags/announcement"},{label:"release",permalink:"/superbenchmark/blog/tags/release"}],readingTime:1.31,truncated:!1,prevItem:{title:"Releasing SuperBench v0.11",permalink:"/superbenchmark/blog/release-sb-v0.11"},nextItem:{title:"Releasing SuperBench v0.9",permalink:"/superbenchmark/blog/release-sb-v0.9"}},c=[{value:"SuperBench 0.10.0 Release Notes",id:"superbench-0100-release-notes",children:[{value:"SuperBench Improvements",id:"superbench-improvements",children:[]},{value:"Micro-benchmark Improvements",id:"micro-benchmark-improvements",children:[]},{value:"Model Benchmark Improvements",id:"model-benchmark-improvements",children:[]},{value:"Result Analysis",id:"result-analysis",children:[]}]}],m={toc:c};function s(e){var t=e.components,r=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,n.Z)({},m,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"We are very happy to announce that ",(0,o.kt)("strong",{parentName:"p"},"SuperBench 0.10.0 version")," is officially released today!"),(0,o.kt)("p",null,"You can install and try superbench by following ",(0,o.kt)("a",{parentName:"p",href:"https://microsoft.github.io/superbenchmark/docs/getting-started/installation"},"Getting Started Tutorial"),"."),(0,o.kt)("h2",{id:"superbench-0100-release-notes"},"SuperBench 0.10.0 Release Notes"),(0,o.kt)("h3",{id:"superbench-improvements"},"SuperBench Improvements"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Support monitoring for AMD GPUs."),(0,o.kt)("li",{parentName:"ul"},"Support ROCm 5.7 and ROCm 6.0 dockerfile."),(0,o.kt)("li",{parentName:"ul"},"Add MSCCL support for Nvidia GPU."),(0,o.kt)("li",{parentName:"ul"},"Fix NUMA domains swap issue in NDv4 topology file."),(0,o.kt)("li",{parentName:"ul"},"Add NDv5 topo file."),(0,o.kt)("li",{parentName:"ul"},"Fix NCCL and NCCL-test to 2.18.3 for hang issue in CUDA 12.2.")),(0,o.kt)("h3",{id:"micro-benchmark-improvements"},"Micro-benchmark Improvements"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Add HPL random generator to gemm-flops with ROCm."),(0,o.kt)("li",{parentName:"ul"},"Add DirectXGPURenderFPS benchmark to measure the FPS of rendering simple frames."),(0,o.kt)("li",{parentName:"ul"},"Add HWDecoderFPS benchmark to measure the FPS of hardware decoder performance."),(0,o.kt)("li",{parentName:"ul"},"Update Docker image for H100 support."),(0,o.kt)("li",{parentName:"ul"},"Update MLC version into 3.10 for CUDA/ROCm dockerfile."),(0,o.kt)("li",{parentName:"ul"},"Bug fix for GPU Burn test."),(0,o.kt)("li",{parentName:"ul"},"Support INT8 in cublaslt function."),(0,o.kt)("li",{parentName:"ul"},"Add hipBLASLt function benchmark."),(0,o.kt)("li",{parentName:"ul"},"Support cpu-gpu and gpu-cpu in ib-validation."),(0,o.kt)("li",{parentName:"ul"},"Support graph mode in NCCL/RCCL benchmarks for latency metrics."),(0,o.kt)("li",{parentName:"ul"},"Support cpp implementation in distributed inference benchmark."),(0,o.kt)("li",{parentName:"ul"},"Add O2 option for gpu copy ROCm build."),(0,o.kt)("li",{parentName:"ul"},"Support different hipblasLt data types in dist inference."),(0,o.kt)("li",{parentName:"ul"},"Support in-place in NCCL/RCCL benchmark."),(0,o.kt)("li",{parentName:"ul"},"Support data type option in NCCL/RCCL benchmark."),(0,o.kt)("li",{parentName:"ul"},"Improve P2P performance with fine-grained GPU memory in GPU-copy test for AMD GPUs."),(0,o.kt)("li",{parentName:"ul"},"Update hipblaslt GEMM metric unit to tflops."),(0,o.kt)("li",{parentName:"ul"},"Support FP8 for hipblaslt benchmark.")),(0,o.kt)("h3",{id:"model-benchmark-improvements"},"Model Benchmark Improvements"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Change torch.distributed.launch to torchrun."),(0,o.kt)("li",{parentName:"ul"},"Support Megatron-LM/Megatron-Deepspeed GPT pretrain benchmark.")),(0,o.kt)("h3",{id:"result-analysis"},"Result Analysis"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Support baseline generation from multiple nodes.")))}s.isMDXComponent=!0}}]);