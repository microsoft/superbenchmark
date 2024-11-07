"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[1777],{5680:(e,r,t)=>{t.d(r,{xA:()=>s,yg:()=>m});var n=t(6540);function a(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function o(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function i(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?o(Object(t),!0).forEach((function(r){a(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function c(e,r){if(null==e)return{};var t,n,a=function(e,r){if(null==e)return{};var t,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)t=o[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(e,r);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)t=o[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var l=n.createContext({}),u=function(e){var r=n.useContext(l),t=r;return e&&(t="function"==typeof e?e(r):i(i({},r),e)),t},s=function(e){var r=u(e.components);return n.createElement(l.Provider,{value:r},e.children)},d={inlineCode:"code",wrapper:function(e){var r=e.children;return n.createElement(n.Fragment,{},r)}},p=n.forwardRef((function(e,r){var t=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,s=c(e,["components","mdxType","originalType","parentName"]),p=u(t),m=a,f=p["".concat(l,".").concat(m)]||p[m]||d[m]||o;return t?n.createElement(f,i(i({ref:r},s),{},{components:t})):n.createElement(f,i({ref:r},s))}));function m(e,r){var t=arguments,a=r&&r.mdxType;if("string"==typeof e||a){var o=t.length,i=new Array(o);i[0]=p;var c={};for(var l in r)hasOwnProperty.call(r,l)&&(c[l]=r[l]);c.originalType=e,c.mdxType="string"==typeof e?e:a,i[1]=c;for(var u=2;u<o;u++)i[u]=t[u];return n.createElement.apply(null,i)}return n.createElement.apply(null,t)}p.displayName="MDXCreateElement"},8613:(e,r,t)=>{t.r(r),t.d(r,{contentTitle:()=>l,default:()=>p,frontMatter:()=>c,metadata:()=>u,toc:()=>s});var n=t(8168),a=t(8587),o=(t(6540),t(5680)),i=["components"],c={id:"introduction"},l="Introduction",u={unversionedId:"introduction",id:"introduction",isDocsHomePage:!1,title:"Introduction",description:"Features",source:"@site/../docs/introduction.md",sourceDirName:".",slug:"/introduction",permalink:"/superbenchmark/docs/introduction",editUrl:"https://github.com/microsoft/superbenchmark/edit/main/website/../docs/introduction.md",version:"current",frontMatter:{id:"introduction"},sidebar:"docs",next:{title:"Installation",permalink:"/superbenchmark/docs/getting-started/installation"}},s=[{value:"Features",id:"features",children:[]},{value:"Overview",id:"overview",children:[]}],d={toc:s};function p(e){var r=e.components,c=(0,a.A)(e,i);return(0,o.yg)("wrapper",(0,n.A)({},d,c,{components:r,mdxType:"MDXLayout"}),(0,o.yg)("h1",{id:"introduction"},"Introduction"),(0,o.yg)("h2",{id:"features"},"Features"),(0,o.yg)("p",null,(0,o.yg)("strong",{parentName:"p"},"SuperBench")," is a validation and profiling tool for AI infrastructure, which supports:"),(0,o.yg)("ul",null,(0,o.yg)("li",{parentName:"ul"},"AI infrastructure validation and diagnosis",(0,o.yg)("ul",{parentName:"li"},(0,o.yg)("li",{parentName:"ul"},"Distributed validation tools to validate hundreds or thousands of servers automatically"),(0,o.yg)("li",{parentName:"ul"},"Consider both raw hardware and E2E model performance with ML workload patterns"),(0,o.yg)("li",{parentName:"ul"},"Build a contract to identify hardware issues"),(0,o.yg)("li",{parentName:"ul"},"Provide infrastructural-oriented criteria as Performance/Quality Gates for hardware and system release"),(0,o.yg)("li",{parentName:"ul"},"Provide detailed performance report and advanced analysis tool"))),(0,o.yg)("li",{parentName:"ul"},"AI workload benchmarking and profiling",(0,o.yg)("ul",{parentName:"li"},(0,o.yg)("li",{parentName:"ul"},"Provide comprehensive performance comparison between different existing hardware"),(0,o.yg)("li",{parentName:"ul"},"Provide insights for hardware and software co-design")))),(0,o.yg)("p",null,"It provides micro-benchmark for primitive computation and communication benchmarking,\nas well as model-benchmark to measure domain-aware end-to-end deep learning workloads."),(0,o.yg)("h2",{id:"overview"},"Overview"),(0,o.yg)("p",null,"The following figure shows the capabilities provided by SuperBench core framework and its extension."),(0,o.yg)("p",null,(0,o.yg)("img",{alt:"SuperBench Structure",src:t(8699).A})))}p.isMDXComponent=!0},8699:(e,r,t)=>{t.d(r,{A:()=>n});const n=t.p+"assets/images/architecture-31a8f49e1763a52efd81fc0fa4bad05b.svg"}}]);