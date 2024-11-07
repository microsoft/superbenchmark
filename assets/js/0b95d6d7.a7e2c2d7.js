"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[382],{5680:(e,n,r)=>{r.d(n,{xA:()=>m,yg:()=>d});var t=r(6540);function a(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function i(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function o(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?i(Object(r),!0).forEach((function(n){a(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function l(e,n){if(null==e)return{};var r,t,a=function(e,n){if(null==e)return{};var r,t,a={},i=Object.keys(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||(a[r]=e[r]);return a}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var p=t.createContext({}),c=function(e){var n=t.useContext(p),r=n;return e&&(r="function"==typeof e?e(n):o(o({},n),e)),r},m=function(e){var n=c(e.components);return t.createElement(p.Provider,{value:n},e.children)},s={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},u=t.forwardRef((function(e,n){var r=e.components,a=e.mdxType,i=e.originalType,p=e.parentName,m=l(e,["components","mdxType","originalType","parentName"]),u=c(r),d=a,h=u["".concat(p,".").concat(d)]||u[d]||s[d]||i;return r?t.createElement(h,o(o({ref:n},m),{},{components:r})):t.createElement(h,o({ref:n},m))}));function d(e,n){var r=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var i=r.length,o=new Array(i);o[0]=u;var l={};for(var p in n)hasOwnProperty.call(n,p)&&(l[p]=n[p]);l.originalType=e,l.mdxType="string"==typeof e?e:a,o[1]=l;for(var c=2;c<i;c++)o[c]=r[c];return t.createElement.apply(null,o)}return t.createElement.apply(null,r)}u.displayName="MDXCreateElement"},5679:(e,n,r)=>{r.r(n),r.d(n,{contentTitle:()=>p,default:()=>u,frontMatter:()=>l,metadata:()=>c,toc:()=>m});var t=r(8168),a=r(8587),i=(r(6540),r(5680)),o=["components"],l={slug:"release-sb-v0.5",title:"Releasing SuperBench v0.5",author:"Peng Cheng",author_title:"SuperBench Team",author_url:"https://github.com/cp5555",author_image_url:"https://github.com/cp5555.png",tags:["superbench","announcement","release"]},p=void 0,c={permalink:"/superbenchmark/blog/release-sb-v0.5",editUrl:"https://github.com/microsoft/superbenchmark/edit/main/website/blog/2022-04-28-release-0-5.md",source:"@site/blog/2022-04-28-release-0-5.md",title:"Releasing SuperBench v0.5",description:"We are very happy to announce that SuperBench 0.5.0 version is officially released today!",date:"2022-04-28T00:00:00.000Z",formattedDate:"April 28, 2022",tags:[{label:"superbench",permalink:"/superbenchmark/blog/tags/superbench"},{label:"announcement",permalink:"/superbenchmark/blog/tags/announcement"},{label:"release",permalink:"/superbenchmark/blog/tags/release"}],readingTime:1.285,truncated:!1,prevItem:{title:"Releasing SuperBench v0.6",permalink:"/superbenchmark/blog/release-sb-v0.6"},nextItem:{title:"Releasing SuperBench v0.4",permalink:"/superbenchmark/blog/release-sb-v0.4"}},m=[{value:"SuperBench 0.5.0 Release Notes",id:"superbench-050-release-notes",children:[{value:"Micro-benchmark Improvements",id:"micro-benchmark-improvements",children:[]},{value:"Model-benchmark Improvements",id:"model-benchmark-improvements",children:[]},{value:"Inference Benchmark Improvements",id:"inference-benchmark-improvements",children:[]},{value:"Other Improvements",id:"other-improvements",children:[]},{value:"Data Diagnosis and Analysis",id:"data-diagnosis-and-analysis",children:[]}]}],s={toc:m};function u(e){var n=e.components,r=(0,a.A)(e,o);return(0,i.yg)("wrapper",(0,t.A)({},s,r,{components:n,mdxType:"MDXLayout"}),(0,i.yg)("p",null,"We are very happy to announce that ",(0,i.yg)("strong",{parentName:"p"},"SuperBench 0.5.0 version")," is officially released today!"),(0,i.yg)("p",null,"You can install and try superbench by following ",(0,i.yg)("a",{parentName:"p",href:"https://microsoft.github.io/superbenchmark/docs/getting-started/installation"},"Getting Started Tutorial"),"."),(0,i.yg)("h2",{id:"superbench-050-release-notes"},"SuperBench 0.5.0 Release Notes"),(0,i.yg)("h3",{id:"micro-benchmark-improvements"},"Micro-benchmark Improvements"),(0,i.yg)("ul",null,(0,i.yg)("li",{parentName:"ul"},"Support NIC only NCCL bandwidth benchmark on single node in NCCL/RCCL bandwidth test."),(0,i.yg)("li",{parentName:"ul"},"Support bi-directional bandwidth benchmark in GPU copy bandwidth test."),(0,i.yg)("li",{parentName:"ul"},"Support data checking in GPU copy bandwidth test."),(0,i.yg)("li",{parentName:"ul"},"Update rccl-tests submodule to fix divide by zero error."),(0,i.yg)("li",{parentName:"ul"},"Add GPU-Burn micro-benchmark.")),(0,i.yg)("h3",{id:"model-benchmark-improvements"},"Model-benchmark Improvements"),(0,i.yg)("ul",null,(0,i.yg)("li",{parentName:"ul"},"Sync results on root rank for e2e model benchmarks in distributed mode."),(0,i.yg)("li",{parentName:"ul"},"Support customized ",(0,i.yg)("inlineCode",{parentName:"li"},"env")," in local and torch.distributed mode."),(0,i.yg)("li",{parentName:"ul"},"Add support for pytorch>=1.9.0."),(0,i.yg)("li",{parentName:"ul"},"Keep BatchNorm as fp32 for pytorch cnn models cast to fp16."),(0,i.yg)("li",{parentName:"ul"},"Remove FP16 samples type converting time."),(0,i.yg)("li",{parentName:"ul"},"Support FAMBench.")),(0,i.yg)("h3",{id:"inference-benchmark-improvements"},"Inference Benchmark Improvements"),(0,i.yg)("ul",null,(0,i.yg)("li",{parentName:"ul"},"Revise the default setting for inference benchmark."),(0,i.yg)("li",{parentName:"ul"},"Add percentile metrics for inference benchmarks."),(0,i.yg)("li",{parentName:"ul"},"Support T4 and A10 in GEMM benchmark."),(0,i.yg)("li",{parentName:"ul"},"Add configuration with inference benchmark.")),(0,i.yg)("h3",{id:"other-improvements"},"Other Improvements"),(0,i.yg)("ul",null,(0,i.yg)("li",{parentName:"ul"},"Add command to support listing all optional parameters for benchmarks."),(0,i.yg)("li",{parentName:"ul"},"Unify benchmark naming convention and support multiple tests with same benchmark and different parameters/options in one configuration file."),(0,i.yg)("li",{parentName:"ul"},"Support timeout to detect the benchmark failure and stop the process automatically."),(0,i.yg)("li",{parentName:"ul"},"Add rocm5.0 dockerfile."),(0,i.yg)("li",{parentName:"ul"},"Improve output interface.")),(0,i.yg)("h3",{id:"data-diagnosis-and-analysis"},"Data Diagnosis and Analysis"),(0,i.yg)("ul",null,(0,i.yg)("li",{parentName:"ul"},"Support multi-benchmark check."),(0,i.yg)("li",{parentName:"ul"},"Support result summary in md, html and excel formats."),(0,i.yg)("li",{parentName:"ul"},"Support data diagnosis in md and html formats."),(0,i.yg)("li",{parentName:"ul"},"Support result output for all nodes in data diagnosis.")))}u.isMDXComponent=!0}}]);