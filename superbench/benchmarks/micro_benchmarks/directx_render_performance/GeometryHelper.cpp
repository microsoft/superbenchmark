
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "GeometryHelper.h"

namespace MathHelper
{
    DirectX::XMFLOAT4X4 Identity4x4()
    {

        DirectX::XMFLOAT4X4 identity;
        DirectX::XMStoreFloat4x4(&identity, DirectX::XMMatrixIdentity());
        return identity;
    }

    float genRand2N_f(int n)
    {
        srand((unsigned int)time(NULL));
        // Seed
        std::random_device rd;

        // Random number generator
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<float> distribution(0, n);
        return distribution(generator);
    }

    uint16_t genRand2N_large(int n)
    {
        srand((unsigned int)time(NULL));
        // Seed
        std::random_device rd;

        // Random number generator
        std::default_random_engine generator(rd());
        std::uniform_int_distribution<unsigned int> distribution(0, 0xFFFFFFFFFFFFFFFF);
        return distribution(generator) % n;
    };
}
