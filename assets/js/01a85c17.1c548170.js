"use strict";(self.webpackChunksuperbench_website=self.webpackChunksuperbench_website||[]).push([[8209],{9937:(e,a,t)=>{t.d(a,{A:()=>b});var r=t(6540),s=t(53),n=t(4676);const l="sidebar_q+wC",i="sidebarItemTitle_9G5K",c="sidebarItemList_6T4b",m="sidebarItem_cjdF",o="sidebarItemLink_zyXk",u="sidebarItemLinkActive_wcJs";var g=t(4798);function b(e){var a=e.sidebar;return 0===a.items.length?null:r.createElement("nav",{className:(0,s.A)(l,"thin-scrollbar"),"aria-label":(0,g.T)({id:"theme.blog.sidebar.navAriaLabel",message:"Blog recent posts navigation",description:"The ARIA label for recent posts in the blog sidebar"})},r.createElement("div",{className:(0,s.A)(i,"margin-bottom--md")},a.title),r.createElement("ul",{className:c},a.items.map((function(e){return r.createElement("li",{key:e.permalink,className:m},r.createElement(n.A,{isNavLink:!0,to:e.permalink,className:o,activeClassName:u},e.title))}))))}},3395:(e,a,t)=>{t.r(a),t.d(a,{default:()=>m});var r=t(6540),s=t(3093),n=t(4676),l=t(9937),i=t(4798),c=t(3008);const m=function(e){var a=e.tags,t=e.sidebar,m=(0,i.T)({id:"theme.tags.tagsPageTitle",message:"Tags",description:"The title of the tag list page"}),o={};Object.keys(a).forEach((function(e){var a=function(e){return e[0].toUpperCase()}(e);o[a]=o[a]||[],o[a].push(e)}));var u=Object.entries(o).sort((function(e,a){var t=e[0],r=a[0];return t.localeCompare(r)})).map((function(e){var t=e[0],s=e[1];return r.createElement("article",{key:t},r.createElement("h2",null,t),s.map((function(e){return r.createElement(n.A,{className:"padding-right--md",href:a[e].permalink,key:e},a[e].name," (",a[e].count,")")})),r.createElement("hr",null))})).filter((function(e){return null!=e}));return r.createElement(s.A,{title:m,wrapperClassName:c.GN.wrapper.blogPages,pageClassName:c.GN.page.blogTagsListPage,searchMetadatas:{tag:"blog_tags_list"}},r.createElement("div",{className:"container margin-vert--lg"},r.createElement("div",{className:"row"},r.createElement("aside",{className:"col col--3"},r.createElement(l.A,{sidebar:t})),r.createElement("main",{className:"col col--7"},r.createElement("h1",null,m),r.createElement("section",{className:"margin-vert--lg"},u)))))}}}]);