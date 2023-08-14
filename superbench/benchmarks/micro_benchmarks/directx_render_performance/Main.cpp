// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RenderGeometryPass.h"
#include "RenderLightingPass.h"
#include "RenderShadowMapPass.h"
#include <codecvt>
#include <cstdio>
#include <iostream>
#include <locale>
#include <string>
#include <tuple>
#include <windows.h>

/*
 * @brief: Main message handler for the sample.
 */
LRESULT WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    // Handle window event.
    switch (message) {
    case WM_CLOSE:
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

/*
 * @brief: Main window procedure.
 */
static LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    // Forward hwnd on because we can get messages (e.g., WM_CREATE)
    // before CreateWindow returns, and thus before mhMainWnd is valid.
    return WindowProc(hwnd, msg, wParam, lParam);
}

/*
 * @brief: Register a window app.
 */
bool InitMainWindow(HINSTANCE hInstance, int width, int height, HWND &hMainWnd, const std::wstring &winTitle,
                    bool quiet_mode) {
    WNDCLASS wc;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = MainWndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(0, IDI_APPLICATION);
    wc.hCursor = LoadCursor(0, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
    wc.lpszMenuName = 0;
    wc.lpszClassName = L"MainWnd";

    if (!RegisterClass(&wc)) {
        return false;
    }

    // Compute window rectangle dimensions based on requested client area dimensions.
    RECT R = {0, 0, width, height};
    AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
    width = R.right - R.left;
    height = R.bottom - R.top;

    hMainWnd = CreateWindow(wc.lpszClassName, winTitle.c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                            width, height, 0, 0, hInstance, 0);
    if (!hMainWnd) {
        return false;
    }

    if (!quiet_mode) {
        ShowWindow(hMainWnd, SW_SHOW);
        UpdateWindow(hMainWnd);
    }
    return true;
}

/*
 * @brief: Load the render microbenchmark according to the pass type.
 */
std::unique_ptr<RenderApp> get_render_pointer(BenchmarkOptions &args, HINSTANCE hInstance, HWND hMainWnd,
                                              std::wstring &winTitle) {
    if (args.m_pass_type == PassType::GeometryPass) {
        return std::make_unique<RenderGeometryPass>(&args, hInstance, hMainWnd, winTitle);
    } else if (args.m_pass_type == PassType::ShadowMapPass) {
        return std::make_unique<RenderShadowMapPass>(&args, hInstance, hMainWnd, winTitle);
    } else if (args.m_pass_type == PassType::LightingPass) {
        return std::make_unique<RenderLightingPass>(&args, hInstance, hMainWnd, winTitle);
    } else
        throw "invalid pass name";
}

/*
 * @brief: Main entry point for a Windows application.
 */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Enable console attach and redirect stdout/stderr to console.
    if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) {
        FILE *stream;
        if (freopen_s(&stream, "CONOUT$", "w", stdout) == 0) {
            printf("Hello, Console!\n");
        }
        if (freopen_s(&stream, "CONOUT$", "w", stderr) == 0) {
            fprintf(stderr, "Hello, Error Console!\n");
        }
        // Or use std::cout
        std::cout << "Hello from std::cout" << std::endl;
    }

    MSG msg = {0};
    try {
        // Parse command line arguments.
        BenchmarkOptions args(__argc, __argv);
        args.init();
        // Create the main window.
        HWND hMainWnd;
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
        std::wstring winTitle = converter.from_bytes("");
        if (!InitMainWindow(hInstance, args.m_width, args.m_height, hMainWnd, winTitle, args.m_quiet))
            return -1;

        // Create the render microbenchmark.
        auto app_sample = get_render_pointer(args, hInstance, hMainWnd, winTitle);
        app_sample->Initialize();
        app_sample->LoadAssets();

        while (msg.message != WM_QUIT) {
            // If there are Window messages then process them.
            // We need to handle message here otherwise it is no response.
            if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                // Update and render per frame.
                app_sample->Tick();
            }
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }

    return (int)msg.wParam;
}
