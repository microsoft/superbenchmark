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
      label: 'User Tutorial',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Benchmarks',
          collapsed: false,
          items: [
            'user-tutorial/benchmarks/micro-benchmarks',
            'user-tutorial/benchmarks/model-benchmarks',
            'user-tutorial/benchmarks/docker-benchmarks',
          ],
        },
        'user-tutorial/system-config',
        'user-tutorial/data-diagnosis',
        'user-tutorial/result-summary',
        'user-tutorial/baseline-generation',
        'user-tutorial/monitor',
        'user-tutorial/container-images',
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
