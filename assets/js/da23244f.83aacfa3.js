"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[2995],{5680:(e,n,t)=>{t.d(n,{xA:()=>s,yg:()=>g});var r=t(6540);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function l(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?l(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):l(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)t=l[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)t=l[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var u=r.createContext({}),p=function(e){var n=r.useContext(u),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},s=function(e){var n=p(e.components);return r.createElement(u.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,l=e.originalType,u=e.parentName,s=i(e,["components","mdxType","originalType","parentName"]),m=p(t),g=a,y=m["".concat(u,".").concat(g)]||m[g]||c[g]||l;return t?r.createElement(y,o(o({ref:n},s),{},{components:t})):r.createElement(y,o({ref:n},s))}));function g(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var l=t.length,o=new Array(l);o[0]=m;var i={};for(var u in n)hasOwnProperty.call(n,u)&&(i[u]=n[u]);i.originalType=e,i.mdxType="string"==typeof e?e:a,o[1]=i;for(var p=2;p<l;p++)o[p]=t[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},147:(e,n,t)=>{t.r(n),t.d(n,{contentTitle:()=>u,default:()=>m,frontMatter:()=>i,metadata:()=>p,toc:()=>s});var r=t(8168),a=t(8587),l=(t(6540),t(5680)),o=["components"],i={slug:"release-sb-v0.2",title:"Releasing SuperBench v0.2",author:"Tingting Qin",author_title:"SuperBench Team",author_url:"https://github.com/TobeyQin",tags:["superbench","announcement","release"]},u=void 0,p={permalink:"/superbenchmark/blog/release-sb-v0.2",editUrl:"https://github.com/microsoft/superbenchmark/edit/main/website/blog/2021-06-28-release-0-2.md",source:"@site/blog/2021-06-28-release-0-2.md",title:"Releasing SuperBench v0.2",description:"We are very happy to announce that SuperBench 0.2.0 version is officially released today!",date:"2021-06-28T00:00:00.000Z",formattedDate:"June 28, 2021",tags:[{label:"superbench",permalink:"/superbenchmark/blog/tags/superbench"},{label:"announcement",permalink:"/superbenchmark/blog/tags/announcement"},{label:"release",permalink:"/superbenchmark/blog/tags/release"}],readingTime:.915,truncated:!1,prevItem:{title:"Releasing SuperBench v0.3",permalink:"/superbenchmark/blog/release-sb-v0.3"},nextItem:{title:"Introduce SuperBench",permalink:"/superbenchmark/blog/intro-sb"}},s=[{value:"SuperBench 0.2.0 Release Notes",id:"superbench-020-release-notes",children:[{value:"SuperBench Framework",id:"superbench-framework",children:[]},{value:"Supported Benchmarks",id:"supported-benchmarks",children:[]},{value:"Examples and Documents",id:"examples-and-documents",children:[]}]}],c={toc:s};function m(e){var n=e.components,t=(0,a.A)(e,o);return(0,l.yg)("wrapper",(0,r.A)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,l.yg)("p",null,"We are very happy to announce that ",(0,l.yg)("strong",{parentName:"p"},"SuperBench 0.2.0 version")," is officially released today!"),(0,l.yg)("p",null,"You can install and try superbench by following ",(0,l.yg)("a",{parentName:"p",href:"https://microsoft.github.io/superbenchmark/docs/getting-started/installation"},"Getting Started Tutorial"),"."),(0,l.yg)("h2",{id:"superbench-020-release-notes"},"SuperBench 0.2.0 Release Notes"),(0,l.yg)("h3",{id:"superbench-framework"},"SuperBench Framework"),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},"Implemented a CLI to provide a command line interface."),(0,l.yg)("li",{parentName:"ul"},"Implemented Runner for nodes control and management."),(0,l.yg)("li",{parentName:"ul"},"Implemented Executor."),(0,l.yg)("li",{parentName:"ul"},"Implemented Benchmark framework.")),(0,l.yg)("h3",{id:"supported-benchmarks"},"Supported Benchmarks"),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},"Supported Micro-benchmarks",(0,l.yg)("ul",{parentName:"li"},(0,l.yg)("li",{parentName:"ul"},"GEMM FLOPS (GFLOPS, TensorCore, cuBLAS, cuDNN)"),(0,l.yg)("li",{parentName:"ul"},"Kernel Launch Time (Kernel_Launch_Event_Time, Kernel_Launch_Wall_Time)"),(0,l.yg)("li",{parentName:"ul"},"Operator Performance (MatMul, Sharding_MatMul)"))),(0,l.yg)("li",{parentName:"ul"},"Supported Model-benchmarks",(0,l.yg)("ul",{parentName:"li"},(0,l.yg)("li",{parentName:"ul"},"CNN models\n(Reference: ",(0,l.yg)("a",{parentName:"li",href:"https://github.com/pytorch/vision/tree/v0.8.0/torchvision/models"},"torchvision models"),")",(0,l.yg)("ul",{parentName:"li"},(0,l.yg)("li",{parentName:"ul"},"ResNet (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152)"),(0,l.yg)("li",{parentName:"ul"},"DenseNet (DenseNet-161, DenseNet-169, DenseNet-201)"),(0,l.yg)("li",{parentName:"ul"},"VGG (VGG-11, VGG-13, VGG-16, VGG-19, VGG11_bn, VGG13_bn, VGG16_bn, VGG19_bn)"),(0,l.yg)("li",{parentName:"ul"},"MNASNet (mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3)"),(0,l.yg)("li",{parentName:"ul"},"AlexNet"),(0,l.yg)("li",{parentName:"ul"},"GoogLeNet"),(0,l.yg)("li",{parentName:"ul"},"Inception_v3"),(0,l.yg)("li",{parentName:"ul"},"mobilenet_v2"),(0,l.yg)("li",{parentName:"ul"},"ResNeXt (resnext50_32x4d, resnext101_32x8d)"),(0,l.yg)("li",{parentName:"ul"},"Wide ResNet (wide_resnet50_2, wide_resnet101_2)"),(0,l.yg)("li",{parentName:"ul"},"ShuffleNet (shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0)"),(0,l.yg)("li",{parentName:"ul"},"SqueezeNet (squeezenet1_0, squeezenet1_1)"))),(0,l.yg)("li",{parentName:"ul"},"LSTM model"),(0,l.yg)("li",{parentName:"ul"},"BERT models (BERT-Base, BERT-Large)"),(0,l.yg)("li",{parentName:"ul"},"GPT-2 model (specify which config)")))),(0,l.yg)("h3",{id:"examples-and-documents"},"Examples and Documents"),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},"Added examples to run benchmarks respectively."),(0,l.yg)("li",{parentName:"ul"},"Tutorial Documents (introduction, getting-started, developer-guides, APIs, benchmarks)."),(0,l.yg)("li",{parentName:"ul"},"Built SuperBench ",(0,l.yg)("a",{parentName:"li",href:"https://aka.ms/superbench/"},"website"),".")))}m.isMDXComponent=!0}}]);