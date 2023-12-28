// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'SuperBench',
  tagline: 'Hardware and Software Benchmarks for AI Systems',
  url: 'https://microsoft.github.io',
  baseUrl: '/superbenchmark/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'microsoft',
  projectName: 'superbenchmark',
  themeConfig: {
    navbar: {
      title: 'SuperBench',
      logo: {
        alt: 'Docusaurus Logo',
        src: 'img/logo.svg',
      },
      items: [
        // left
        {
          type: 'doc',
          docId: 'introduction',
          label: 'Docs',
          position: 'left',
        },
        {
          type: 'doc',
          docId: 'cli',
          label: 'API',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        // right
        {
          href: 'https://github.com/microsoft/superbenchmark',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: '/docs/introduction',
            },
            {
              label: 'Getting Started',
              to: '/docs/getting-started/installation',
            },
            {
              label: 'API',
              to: '/docs/cli',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Issues',
              href: 'https://github.com/microsoft/superbenchmark/issues',
            },
            {
              label: 'Discussion',
              href: 'https://github.com/microsoft/superbenchmark/discussions',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/microsoft/superbenchmark',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} SuperBench. <br> Built with Docusaurus and hosted by GitHub.`,
    },
    announcementBar: {
      id: 'supportus',
      content:
        '📢 <a href="https://microsoft.github.io/superbenchmark/blog/release-sb-v0.10">v0.10.0</a> has been released! ' +
        '⭐️ If you like SuperBench, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/microsoft/superbenchmark">GitHub</a>! ⭐️',
    },
    algolia: {
      apiKey: '6809111d3dabf59fe562601d591d7c53',
      indexName: 'superbenchmark',
      contextualSearch: true,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['ini'],
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: '../docs',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/microsoft/superbenchmark/edit/main/website/',
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/microsoft/superbenchmark/edit/main/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/index.css'),
        },
      },
    ],
  ],
};
