"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[2349],{5680:(e,t,r)=>{r.d(t,{xA:()=>p,yg:()=>y});var n=r(6540);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function l(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?l(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):l(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function g(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},l=Object.keys(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var o=n.createContext({}),u=function(e){var t=n.useContext(o),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},p=function(e){var t=u(e.components);return n.createElement(o.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},c=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,l=e.originalType,o=e.parentName,p=g(e,["components","mdxType","originalType","parentName"]),c=u(r),y=a,d=c["".concat(o,".").concat(y)]||c[y]||m[y]||l;return r?n.createElement(d,i(i({ref:t},p),{},{components:r})):n.createElement(d,i({ref:t},p))}));function y(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=r.length,i=new Array(l);i[0]=c;var g={};for(var o in t)hasOwnProperty.call(t,o)&&(g[o]=t[o]);g.originalType=e,g.mdxType="string"==typeof e?e:a,i[1]=g;for(var u=2;u<l;u++)i[u]=r[u];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}c.displayName="MDXCreateElement"},6982:(e,t,r)=>{r.d(t,{A:()=>a});var n=r(6540);const a=function(e){var t=e.children,r=e.hidden,a=e.className;return n.createElement("div",{role:"tabpanel",hidden:r,className:a},t)}},2689:(e,t,r)=>{r.d(t,{A:()=>p});var n=r(6540),a=r(4879),l=r(53);const i="tabItem_vU9c",g="tabItemActive_cw6a";var o=37,u=39;const p=function(e){var t=e.lazy,r=e.block,p=e.defaultValue,m=e.values,c=e.groupId,y=e.className,d=(0,a.A)(),s=d.tabGroupChoices,h=d.setTabGroupChoices,v=(0,n.useState)(p),N=v[0],b=v[1],f=n.Children.toArray(e.children),w=[];if(null!=c){var C=s[c];null!=C&&C!==N&&m.some((function(e){return e.value===C}))&&b(C)}var O=function(e){var t=e.currentTarget,r=w.indexOf(t),n=m[r].value;b(n),null!=c&&(h(c,n),setTimeout((function(){var e,r,n,a,l,i,o,u;(e=t.getBoundingClientRect(),r=e.top,n=e.left,a=e.bottom,l=e.right,i=window,o=i.innerHeight,u=i.innerWidth,r>=0&&l<=u&&a<=o&&n>=0)||(t.scrollIntoView({block:"center",behavior:"smooth"}),t.classList.add(g),setTimeout((function(){return t.classList.remove(g)}),2e3))}),150))},S=function(e){var t,r;switch(e.keyCode){case u:var n=w.indexOf(e.target)+1;r=w[n]||w[0];break;case o:var a=w.indexOf(e.target)-1;r=w[a]||w[w.length-1]}null==(t=r)||t.focus()};return n.createElement("div",{className:"tabs-container"},n.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,l.A)("tabs",{"tabs--block":r},y)},m.map((function(e){var t=e.value,r=e.label;return n.createElement("li",{role:"tab",tabIndex:N===t?0:-1,"aria-selected":N===t,className:(0,l.A)("tabs__item",i,{"tabs__item--active":N===t}),key:t,ref:function(e){return w.push(e)},onKeyDown:S,onFocus:O,onClick:O},r)}))),t?(0,n.cloneElement)(f.filter((function(e){return e.props.value===N}))[0],{className:"margin-vert--md"}):n.createElement("div",{className:"margin-vert--md"},f.map((function(e,t){return(0,n.cloneElement)(e,{key:t,hidden:e.props.value!==N})}))))}},8555:(e,t,r)=>{r.d(t,{A:()=>n});const n=(0,r(6540).createContext)(void 0)},4879:(e,t,r)=>{r.d(t,{A:()=>l});var n=r(6540),a=r(8555);const l=function(){var e=(0,n.useContext)(a.A);if(null==e)throw new Error('"useUserPreferencesContext" is used outside of "Layout" component.');return e}},6524:(e,t,r)=>{r.r(t),r.d(t,{contentTitle:()=>p,default:()=>d,frontMatter:()=>u,metadata:()=>m,toc:()=>c});var n=r(8168),a=r(8587),l=(r(6540),r(5680)),i=r(2689),g=r(6982),o=["components"],u={id:"container-images"},p="Container Images",m={unversionedId:"user-tutorial/container-images",id:"user-tutorial/container-images",isDocsHomePage:!1,title:"Container Images",description:"SuperBench provides a set of OCI-compliant container images,",source:"@site/../docs/user-tutorial/container-images.mdx",sourceDirName:"user-tutorial",slug:"/user-tutorial/container-images",permalink:"/superbenchmark/docs/user-tutorial/container-images",editUrl:"https://github.com/microsoft/superbenchmark/edit/main/website/../docs/user-tutorial/container-images.mdx",version:"current",frontMatter:{id:"container-images"},sidebar:"docs",previous:{title:"Monitor",permalink:"/superbenchmark/docs/user-tutorial/monitor"},next:{title:"Development",permalink:"/superbenchmark/docs/developer-guides/development"}},c=[{value:"Stable tagged versions",id:"stable-tagged-versions",children:[]}],y={toc:c};function d(e){var t=e.components,r=(0,a.A)(e,o);return(0,l.yg)("wrapper",(0,n.A)({},y,r,{components:t,mdxType:"MDXLayout"}),(0,l.yg)("h1",{id:"container-images"},"Container Images"),(0,l.yg)("p",null,"SuperBench provides a set of OCI-compliant container images,\nwhich are hosted on both ",(0,l.yg)("a",{parentName:"p",href:"https://hub.docker.com/r/superbench/superbench/tags"},"Docker Hub"),"\nand ",(0,l.yg)("a",{parentName:"p",href:"https://github.com/microsoft/superbenchmark/pkgs/container/superbenchmark%2Fsuperbench"},"GitHub Container Registry"),"."),(0,l.yg)("p",null,"You can use SuperBench image by ",(0,l.yg)("inlineCode",{parentName:"p"},"superbench/superbench:${tag}")," or ",(0,l.yg)("inlineCode",{parentName:"p"},"ghcr.io/microsoft/superbenchmark/superbench:${tag}"),",\navailable tags are listed below for all stable versions."),(0,l.yg)("h2",{id:"stable-tagged-versions"},"Stable tagged versions"),(0,l.yg)(i.A,{groupId:"gpu-platform",defaultValue:"cuda",values:[{label:"CUDA",value:"cuda"},{label:"ROCm",value:"rocm"},{label:"DirectX",value:"directx"}],mdxType:"Tabs"},(0,l.yg)(g.A,{value:"cuda",mdxType:"TabItem"},(0,l.yg)("table",null,(0,l.yg)("thead",{parentName:"table"},(0,l.yg)("tr",{parentName:"thead"},(0,l.yg)("th",{parentName:"tr",align:null},"Tag"),(0,l.yg)("th",{parentName:"tr",align:null},"Description"))),(0,l.yg)("tbody",{parentName:"table"},(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.11.0-cuda12.4"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.11.0 with CUDA 12.4")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.11.0-cuda12.2"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.11.0 with CUDA 12.2")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.11.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.11.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.10.0-cuda12.2"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.10.0 with CUDA 12.2")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.10.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.10.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.9.0-cuda12.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.9.0 with CUDA 12.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.9.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.9.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.8.0-cuda12.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.8.0 with CUDA 12.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.8.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.8.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.7.0-cuda11.8"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.7.0 with CUDA 11.8")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.7.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.7.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.6.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.6.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.5.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.5.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.4.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.4.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.3.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.3.0 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.2.1-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.2.1 with CUDA 11.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.2.0-cuda11.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.2.0 with CUDA 11.1.1"))))),(0,l.yg)(g.A,{value:"rocm",mdxType:"TabItem"},(0,l.yg)("table",null,(0,l.yg)("thead",{parentName:"table"},(0,l.yg)("tr",{parentName:"thead"},(0,l.yg)("th",{parentName:"tr",align:null},"Tag"),(0,l.yg)("th",{parentName:"tr",align:null},"Description"))),(0,l.yg)("tbody",{parentName:"table"},(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.11.0-rocm6.2"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.11.0 with ROCm 6.2")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.11.0-rocm6.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.11.0 with ROCm 6.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.10.0-rocm6.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.10.0 with ROCm 6.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.10.0-rocm5.7"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.10.0 with ROCm 5.7")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.9.0-rocm5.1.3"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.9.0 with ROCm 5.1.3")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.9.0-rocm5.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.9.0 with ROCm 5.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.9.0-rocm5.0.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.9.0 with ROCm 5.0.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.9.0-rocm5.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.9.0 with ROCm 5.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.8.0-rocm5.1.3"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.8.0 with ROCm 5.1.3")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.8.0-rocm5.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.8.0 with ROCm 5.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.8.0-rocm5.0.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.8.0 with ROCm 5.0.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.8.0-rocm5.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.8.0 with ROCm 5.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.7.0-rocm5.1.3"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.7.0 with ROCm 5.1.3")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.7.0-rocm5.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.7.0 with ROCm 5.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.7.0-rocm5.0.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.7.0 with ROCm 5.0.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.7.0-rocm5.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.7.0 with ROCm 5.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.6.0-rocm5.1.3"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.6.0 with ROCm 5.1.3")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.6.0-rocm5.1.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.6.0 with ROCm 5.1.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.6.0-rocm5.0.1"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.6.0 with ROCm 5.0.1")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.6.0-rocm5.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.6.0 with ROCm 5.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.5.0-rocm5.0.1-pytorch1.9.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.5.0 with ROCm 5.0.1, PyTorch 1.9.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.5.0-rocm5.0-pytorch1.9.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.5.0 with ROCm 5.0, PyTorch 1.9.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.5.0-rocm4.2-pytorch1.7.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.5.0 with ROCm 4.2, PyTorch 1.7.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.5.0-rocm4.0-pytorch1.7.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.5.0 with ROCm 4.0, PyTorch 1.7.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.4.0-rocm4.2-pytorch1.7.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.4.0 with ROCm 4.2, PyTorch 1.7.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.4.0-rocm4.0-pytorch1.7.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.4.0 with ROCm 4.0, PyTorch 1.7.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.3.0-rocm4.2-pytorch1.7.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.3.0 with ROCm 4.2, PyTorch 1.7.0")),(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.3.0-rocm4.0-pytorch1.7.0"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.3.0 with ROCm 4.0, PyTorch 1.7.0"))))),(0,l.yg)(g.A,{value:"directx",mdxType:"TabItem"},(0,l.yg)("table",null,(0,l.yg)("thead",{parentName:"table"},(0,l.yg)("tr",{parentName:"thead"},(0,l.yg)("th",{parentName:"tr",align:null},"Tag"),(0,l.yg)("th",{parentName:"tr",align:null},"Description"))),(0,l.yg)("tbody",{parentName:"table"},(0,l.yg)("tr",{parentName:"tbody"},(0,l.yg)("td",{parentName:"tr",align:null},"v0.9.0-directx12"),(0,l.yg)("td",{parentName:"tr",align:null},"SuperBench v0.9.0 with DirectX12, Windows10-2004")))))))}d.isMDXComponent=!0},53:(e,t,r)=>{function n(e){var t,r,a="";if("string"==typeof e||"number"==typeof e)a+=e;else if("object"==typeof e)if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(r=n(e[t]))&&(a&&(a+=" "),a+=r);else for(t in e)e[t]&&(a&&(a+=" "),a+=t);return a}function a(){for(var e,t,r=0,a="";r<arguments.length;)(e=arguments[r++])&&(t=n(e))&&(a&&(a+=" "),a+=t);return a}r.d(t,{A:()=>a})}}]);