// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.


module.exports = {
  docs: [
    'introduction',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/configuration',
        'getting-started/run-superbench',
      ],
    },
    {
      type: 'category',
      label: 'Tutorial',
      collapsed: false,
      items: [
        'tutorial/container-images',
      ],
    },
    {
      type: 'category',
      label: 'Benchmarks',
      items: [
        'benchmarks/micro-benchmarks',
        'benchmarks/model-benchmarks',
      ],
    },
    {
      type: 'category',
      label: 'Developer Guides',
      items: [
        'developer-guides/development',
        'developer-guides/using-docker',
        'developer-guides/contributing',
      ],
    },
    {
      type: 'category',
      label: 'Design Docs',
      items: [
        'design-docs/overview',
        'design-docs/benchmarks',
      ],
    },
  ],
  api: [
    'cli',
    'superbench-config',
  ],
};
