// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'SuperBench',
  tagline: 'Hardware and Software Benchmarks for AI Systems',
  url: 'https://superbench.xonez.cn',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'alephpiece',
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
          href: 'https://github.com/alephpiece/superbenchmark',
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
              href: 'https://github.com/alephpiece/superbenchmark/issues',
            },
            {
              label: 'Discussion',
              href: 'https://github.com/alephpiece/superbenchmark/discussions',
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
              href: 'https://github.com/alephpiece/superbenchmark',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} SuperBench. <br> Built with Docusaurus and hosted by GitHub.`,
    },
    announcementBar: {
      id: 'supportus',
      content: 'This site is published from a fork of the official SuperBench project.',
    },
    algolia: {
      appId: 'WKHUVH720Q',
      apiKey: '0b3312f228b10e7a9ba0d4e022277790',
      indexName: 'superbench',
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
          editUrl: 'https://github.com/alephpiece/superbenchmark/edit/dtk/',
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/alephpiece/superbenchmark/edit/dtk/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/index.css'),
        },
      },
    ],
  ],
};
