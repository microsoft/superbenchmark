// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <codecvt>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../directx_utils/Options.h"

using namespace std;

// enum class for pass type
enum class PassType { GeometryPass, ShadowMapPass, LightingPass };

class BenchmarkOptions : public Options {
  public:
    int m_textureSize = 0;
    int m_textureNum = 10;
    int m_vertexNum = 3000;
    int m_indexNum = 3000;
    int m_width = 1080;
    int m_height = 720;
    int m_warmup = 500;
    int m_num_object = 1;
    string m_outfile = "outfile.txt";
    PassType m_pass_type = PassType::ShadowMapPass;
    int m_num_frames = 3000;
    int m_num_light = 1;
    bool m_quiet = true;

    BenchmarkOptions(int argc, char *argv[]) : Options(argc, argv) {}

    virtual void get_option_usage() {
        cout << "Usage: " << endl;
        cout << "  --width <int>        set the width of the window" << endl;
        cout << "  --height <int>       set the height of the window" << endl;
        cout << "  --warmup <int>       set the warmup frames" << endl;
        cout << "  --vertex <int>       set the number of vertices" << endl;
        cout << "  --index <int>        set the number of indices" << endl;
        cout << "  --texture_size <int> set the size of textures <x,x>" << endl;
        cout << "  --outfile <string>   set the output file name" << endl;
        cout << "  --pass <string>      set the pass type" << endl;
        cout << "  --object <int>       set the number of objects" << endl;
        cout << "  --frame <int>        set the number of frames" << endl;
        cout << "  --light <int>        set the number of lights" << endl;
        cout << "  --quiet              disable window" << endl;
    }

    virtual void parse_arguments() {
        m_width = get_cmd_line_argument_int("--width", 1080);
        m_height = get_cmd_line_argument_int("--height", 720);
        m_warmup = get_cmd_line_argument_int("--warmup", 500);
        m_vertexNum = get_cmd_line_argument_int("--vertex", m_vertexNum);
        m_indexNum = get_cmd_line_argument_int("--index", m_indexNum);
        m_textureSize = get_cmd_line_argument_int("--texture", 3);
        m_textureNum = get_cmd_line_argument_int("--texture_num", 3);
        m_outfile = get_cmd_line_argument_string("--outfile");
        auto pass = get_cmd_line_argument_string("--pass");
        std::transform(pass.begin(), pass.end(), pass.begin(), [](unsigned char c) { return std::tolower(c); });
        if (pass == "geometry") {
            m_pass_type = PassType::GeometryPass;
        } else if (pass == "shadow") {
            m_pass_type = PassType::ShadowMapPass;
        } else if (pass == "lighting") {
            m_pass_type = PassType::LightingPass;
        } else {
            cout << "Error: Invalid pass type: " << pass << endl;
            exit(1);
        }
        m_num_object = get_cmd_line_argument_int("--object", m_num_object);
        m_num_frames = get_cmd_line_argument_int("--frame", m_num_frames);
        m_num_light = get_cmd_line_argument_int("--light", m_num_light);
        m_quiet = get_cmd_line_argument_bool("--quiet");
    };
};
