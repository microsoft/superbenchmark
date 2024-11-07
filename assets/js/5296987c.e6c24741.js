"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[6326],{5680:(e,t,n)=>{n.d(t,{xA:()=>u,yg:()=>d});var r=n(6540);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function p(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var i=r.createContext({}),s=function(e){var t=r.useContext(i),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},u=function(e){var t=s(e.components);return r.createElement(i.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,i=e.parentName,u=p(e,["components","mdxType","originalType","parentName"]),m=s(n),d=o,g=m["".concat(i,".").concat(d)]||m[d]||c[d]||a;return n?r.createElement(g,l(l({ref:t},u),{},{components:n})):r.createElement(g,l({ref:t},u))}));function d(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,l=new Array(a);l[0]=m;var p={};for(var i in t)hasOwnProperty.call(t,i)&&(p[i]=t[i]);p.originalType=e,p.mdxType="string"==typeof e?e:o,l[1]=p;for(var s=2;s<a;s++)l[s]=n[s];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},8847:(e,t,n)=>{n.r(t),n.d(t,{contentTitle:()=>i,default:()=>m,frontMatter:()=>p,metadata:()=>s,toc:()=>u});var r=n(8168),o=n(8587),a=(n(6540),n(5680)),l=["components"],p={id:"development"},i="Development",s={unversionedId:"developer-guides/development",id:"developer-guides/development",isDocsHomePage:!1,title:"Development",description:"If you want to develop new feature, please follow below steps to set up development environment.",source:"@site/../docs/developer-guides/development.md",sourceDirName:"developer-guides",slug:"/developer-guides/development",permalink:"/superbenchmark/docs/developer-guides/development",editUrl:"https://github.com/microsoft/superbenchmark/edit/main/website/../docs/developer-guides/development.md",version:"current",frontMatter:{id:"development"},sidebar:"docs",previous:{title:"Container Images",permalink:"/superbenchmark/docs/user-tutorial/container-images"},next:{title:"Using Docker",permalink:"/superbenchmark/docs/developer-guides/using-docker"}},u=[{value:"Check Environment",id:"check-environment",children:[]},{value:"Set Up",id:"set-up",children:[]},{value:"Lint and Test",id:"lint-and-test",children:[]},{value:"Submit a Pull Request",id:"submit-a-pull-request",children:[]}],c={toc:u};function m(e){var t=e.components,n=(0,o.A)(e,l);return(0,a.yg)("wrapper",(0,r.A)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,a.yg)("h1",{id:"development"},"Development"),(0,a.yg)("p",null,"If you want to develop new feature, please follow below steps to set up development environment."),(0,a.yg)("p",null,"We suggest you to use ",(0,a.yg)("a",{parentName:"p",href:"https://vscode.github.com/"},"Visual Studio Code")," and install the recommended extensions for this project.\nYou can also develop online with ",(0,a.yg)("a",{parentName:"p",href:"https://github.com/codespaces"},"GitHub Codespaces"),"."),(0,a.yg)("h2",{id:"check-environment"},"Check Environment"),(0,a.yg)("p",null,"Follow ",(0,a.yg)("a",{parentName:"p",href:"/superbenchmark/docs/getting-started/installation"},"System Requirements"),"."),(0,a.yg)("h2",{id:"set-up"},"Set Up"),(0,a.yg)("pre",null,(0,a.yg)("code",{parentName:"pre",className:"language-bash"},"git clone --recurse-submodules -j8 https://github.com/microsoft/superbenchmark\ncd superbenchmark\n\npython3 -m pip install -e .[develop]\n")),(0,a.yg)("h2",{id:"lint-and-test"},"Lint and Test"),(0,a.yg)("p",null,"Format code using yapf."),(0,a.yg)("pre",null,(0,a.yg)("code",{parentName:"pre",className:"language-bash"},"python3 setup.py format\n")),(0,a.yg)("p",null,"Check code style with mypy and flake8"),(0,a.yg)("pre",null,(0,a.yg)("code",{parentName:"pre",className:"language-bash"},"python3 setup.py lint\n")),(0,a.yg)("p",null,"Run unit tests."),(0,a.yg)("pre",null,(0,a.yg)("code",{parentName:"pre",className:"language-bash"},"python3 setup.py test\n")),(0,a.yg)("h2",{id:"submit-a-pull-request"},"Submit a Pull Request"),(0,a.yg)("p",null,"Please install ",(0,a.yg)("inlineCode",{parentName:"p"},"pre-commit")," before ",(0,a.yg)("inlineCode",{parentName:"p"},"git commit")," to run all pre-checks."),(0,a.yg)("pre",null,(0,a.yg)("code",{parentName:"pre",className:"language-bash"},"pre-commit install --install-hooks\n")),(0,a.yg)("p",null,"Open a pull request to main branch on GitHub."))}m.isMDXComponent=!0}}]);