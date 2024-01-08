"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[7920],{3905:function(e,t,r){r.d(t,{Zo:function(){return p},kt:function(){return d}});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var c=n.createContext({}),u=function(e){var t=n.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},p=function(e){var t=u(e.components);return n.createElement(c.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},s=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,c=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),s=u(r),d=a,h=s["".concat(c,".").concat(d)]||s[d]||m[d]||o;return r?n.createElement(h,i(i({ref:t},p),{},{components:r})):n.createElement(h,i({ref:t},p))}));function d(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=s;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var u=2;u<o;u++)i[u]=r[u];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}s.displayName="MDXCreateElement"},599:function(e,t,r){r.r(t),r.d(t,{contentTitle:function(){return c},default:function(){return s},frontMatter:function(){return l},metadata:function(){return u},toc:function(){return p}});var n=r(7462),a=r(3366),o=(r(7294),r(3905)),i=["components"],l={slug:"release-sb-v0.9",title:"Releasing SuperBench v0.9",author:"Peng Cheng",author_title:"SuperBench Team",author_url:"https://github.com/cp5555",author_image_url:"https://github.com/cp5555.png",tags:["superbench","announcement","release"]},c=void 0,u={permalink:"/superbenchmark/blog/release-sb-v0.9",editUrl:"https://github.com/microsoft/superbenchmark/edit/main/website/blog/2023-07-25-release-0-9.md",source:"@site/blog/2023-07-25-release-0-9.md",title:"Releasing SuperBench v0.9",description:"We are very happy to announce that SuperBench 0.9.0 version is officially released today!",date:"2023-07-25T00:00:00.000Z",formattedDate:"July 25, 2023",tags:[{label:"superbench",permalink:"/superbenchmark/blog/tags/superbench"},{label:"announcement",permalink:"/superbenchmark/blog/tags/announcement"},{label:"release",permalink:"/superbenchmark/blog/tags/release"}],readingTime:.89,truncated:!1,prevItem:{title:"Releasing SuperBench v0.10",permalink:"/superbenchmark/blog/release-sb-v0.10"},nextItem:{title:"Releasing SuperBench v0.8",permalink:"/superbenchmark/blog/release-sb-v0.8"}},p=[{value:"SuperBench 0.9.0 Release Notes",id:"superbench-090-release-notes",children:[{value:"SuperBench Improvement",id:"superbench-improvement",children:[]},{value:"Micro-benchmark Improvement",id:"micro-benchmark-improvement",children:[]},{value:"Model Benchmark Improvement",id:"model-benchmark-improvement",children:[]},{value:"Documentation",id:"documentation",children:[]}]}],m={toc:p};function s(e){var t=e.components,r=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,n.Z)({},m,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"We are very happy to announce that ",(0,o.kt)("strong",{parentName:"p"},"SuperBench 0.9.0 version")," is officially released today!"),(0,o.kt)("p",null,"You can install and try superbench by following ",(0,o.kt)("a",{parentName:"p",href:"https://microsoft.github.io/superbenchmark/docs/getting-started/installation"},"Getting Started Tutorial"),"."),(0,o.kt)("h2",{id:"superbench-090-release-notes"},"SuperBench 0.9.0 Release Notes"),(0,o.kt)("h3",{id:"superbench-improvement"},"SuperBench Improvement"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Support Ctrl+C and interrupt to stop all SuperBench testing."),(0,o.kt)("li",{parentName:"ul"},"Support Windows Docker for VDI/Gaming GPU."),(0,o.kt)("li",{parentName:"ul"},"Support DirectX platform for Nvidia and AMD GPU."),(0,o.kt)("li",{parentName:"ul"},"Add System Config Info feature in SB runner to support distributed collection."),(0,o.kt)("li",{parentName:"ul"},"Support DirectX test pipeline.")),(0,o.kt)("h3",{id:"micro-benchmark-improvement"},"Micro-benchmark Improvement"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Add DirectXGPUCopyBw Benchmark to measure HtoD/DtoH bandwidth by DirectX."),(0,o.kt)("li",{parentName:"ul"},"Add DirectXGPUCoreFLops Benchmark to measure peak FLOPS by DirectX.."),(0,o.kt)("li",{parentName:"ul"},"Add DirectXGPUMemBw Benchmark to measure GPU memory bandwidth by DirectX.."),(0,o.kt)("li",{parentName:"ul"},"Add DirectXVCNEncodingLatency Benchmark to measure the VCN hardware encoding latency on AMD graphic GPUs."),(0,o.kt)("li",{parentName:"ul"},"Support best algorithm selection in cudnn-function microbenchmark."),(0,o.kt)("li",{parentName:"ul"},"Revise step time collection in distributed inference benchmark.")),(0,o.kt)("h3",{id:"model-benchmark-improvement"},"Model Benchmark Improvement"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Fix early stop logic due to num_steps in model benchmarks."),(0,o.kt)("li",{parentName:"ul"},"Support TensorRT models on Nvidia H100.")),(0,o.kt)("h3",{id:"documentation"},"Documentation"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Improve documentation for System Config Info."),(0,o.kt)("li",{parentName:"ul"},"Update outdate references.")))}s.isMDXComponent=!0}}]);