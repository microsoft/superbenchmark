// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// IB validation tool is a tool to validate IB traffic of different pattern in multi nodes flexibly
// input：
//  cmd_prefix: the prefix of command to run
//  input_config: the path of input config file, the format of config file is as the following,
//    each row will run in parallel, different rows will run in sequence, each pair is (server index, client index)
//    1,0;2,0;3,0
//    0,1;2,1;3,1
//    0,2;1,2;3,2
//    0,3;1,3;2,3
//  output_path: the path of output csv file

#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <mpi.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

using namespace std;

#define ROOT_RANK 0

int g_world_size;
int g_world_rank;
char g_processor_name[MPI_MAX_PROCESSOR_NAME];

// The struct to store command line arguments
struct Args {
    // The prefix of command to run
    std::string cmd_prefix;
    // The path of input config file
    std::string input_config;
    // The path of output csv file
    std::string output_path;
};

// Parse and save command line arguments
void load_args(int argc, char *argv[], Args &args) {
    // Get and parse command line arguments
    boost::program_options::options_description opt("all options");
    opt.add_options()("cmd_prefix,c",
                      boost::program_options::value<std::string>(&args.cmd_prefix)
                          ->default_value("ib_write_bw -s 33554432 -d ibP257p0s0"),
                      "ib command prefix")(
        "input_config,i", boost::program_options::value<std::string>(&args.input_config)->default_value("config.txt"),
        "the path of input config file")(
        "output_path,o", boost::program_options::value<std::string>(&args.output_path)->default_value("result.csv"),
        "custom the path of the output csv file")("help", "print help info");

    boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, opt), vm);
    boost::program_options::notify(vm);
    if (vm.count("help")) {
        if (g_world_rank == ROOT_RANK)
            std::cout << opt << std::endl;
        return;
    }
    if (g_world_rank == ROOT_RANK) {
        printf("The predix of cmd to run is: %s\n", args.cmd_prefix.c_str());
        printf("Load the config file from: %s\n", args.input_config.c_str());
        printf("Output will be saved to: %s\n", args.output_path.c_str());
    }
}

// Joint vector<string> to vector<char> for mpi to send ips, each string split by '\0'.
char *string_to_char(const std::vector<std::string> &strings) {
    // the max length of each ip is 16
    int max_len = 16;
    // calculate size and allocate memory to save ips for all ranks
    int size = (max_len + 1) * g_world_size + 1;
    char *all_str = (char *)malloc(size * sizeof(char));
    memset(all_str, '\0', size + 1);
    // joint each string in the vector to a long char*
    char *str_idx = all_str;
    for (int i = 0; i < strings.size(); i++) {
        memcpy(str_idx, strings[i].c_str(), (strings[i]).size());
        str_idx = str_idx + max_len + 1;
    }
    return all_str;
}

// Load and parse config file to vector
vector<vector<std::pair<int, int>>> load_config(string filename = "config.txt") {
    // read contents from file
    vector<std::string> config;
    ifstream in(filename);
    string line;

    if (in) {
        while (getline(in, line)) {
            config.push_back(line);
        }
    } else {
        throw exception("Failed to open config file.");
    }

    if (g_world_rank == ROOT_RANK) {
        std::cout << "config: " << std::endl;
        for (const std::string &line : config) {
            std::cout << line << std::endl;
        }
    }

    // parse the string contents to vector
    vector<vector<std::pair<int, int>>> run_in_total;
    try {
        for (const auto &single_line : config) {
            // parse each line like "1,2;2,3;3,4" to vector<std::pair<int, int>>
            vector<string> run_in_parallel;
            vector<std::pair<int, int>> run_pairs_in_parallel;
            // split line to pair by ";"
            boost::split(run_in_parallel, single_line, boost::is_any_of(";"), boost::token_compress_on);
            for (const auto &pair : run_in_parallel) {
                // split pair by ","
                int quote = pair.find(',');
                if (quote == pair.npos) {
                    throw exception("Invalid config format.");
                }
                int first = stoi(pair.substr(0, quote));
                int second = stoi(pair.substr(quote + 1));
                run_pairs_in_parallel.emplace_back(first, second);
            }
            run_in_total.emplace_back(run_pairs_in_parallel);
        }
    } catch (...) {
        throw exception("Invalid config format.");
    }

    return run_in_total;
}

// Execute shell cmd in termial and get output
std::string exec(const char *cmd) {
    char buffer[128];
    std::string result = "";
    // use pipe to execute command
    FILE *pipe = popen(cmd, "r");
    if (!pipe)
        throw exception("popen() failed!");
    try {
        // use buffer to get output
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (const std::exception &e) {
        pclose(pipe);
        throw e;
    }
    pclose(pipe);
    return result;
}

// Get private ip of current machine
std::string get_my_ip() {
    std::string ip;
    string command = "hostname -I | awk '{print $1}'";
    ip = exec(command.c_str());
    int r = ip.find('\n');
    ip = ip.substr(0, r);
    return ip;
}

// Use socket to get a free port
int get_available_port() {
    struct sockaddr_in server_addr;
    int sockfd;
    // open a socket
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        throw exception("Error: failed to open socket.");
    }
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = 0;
    inet_aton("127.0.0.1", &server_addr.sin_addr);
    bzero((char *)&server_addr, sizeof(server_addr));
    // assign a port number
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        throw exception("Error: failed to bind socker.");
    }
    socklen_t len_inet = sizeof(server_addr);
    // get socket name
    if (getsockname(sockfd, (struct sockaddr *)&server_addr, &len_inet) < 0) {
        throw exception("Error: failed to get socketname.");
    }
    // get the port number
    int local_port = ntohs(server_addr.sin_port);
    // make the port can be reused
    int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
        throw exception("Error: failed to set socket.");
    }
    return local_port;
}

// Get and broadcast ports for each run
vector<int> prepare_ports(const vector<std::pair<int, int>> &run_pairs_in_parallel) {
    int pair_count = run_pairs_in_parallel.size();
    vector<int> ports(pair_count);

    for (int index = 0; index < run_pairs_in_parallel.size(); index++) {
        int server_index = run_pairs_in_parallel[index].first;
        // server get a free port and send to rank ROOT_RANK
        if (server_index == g_world_rank) {
            int port = get_available_port();
            MPI_Send(&port, 1, MPI_INT, ROOT_RANK, index, MPI_COMM_WORLD);
        }
        // rank ROOT_RANK recv port from server
        if (g_world_rank == ROOT_RANK) {
            int port;
            MPI_Recv(&port, 1, MPI_INT, server_index, index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ports[index] = port;
        }
    }
    // rank ROOT_RANK broadcast ports to all ranks
    MPI_Bcast(ports.data(), pair_count, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    return ports;
}

// Get and gather ips for all ranks
void gather_ips(vector<string> &ips) {
    // get my ip
    string ip = get_my_ip();
    // convert my ip address from string to vector<char>
    int my_len = ip.size();
    int max_len = 16;
    char my_str_padded[max_len + 1];
    memset(my_str_padded, '\0', max_len + 1);
    memcpy(my_str_padded, ip.c_str(), my_len);
    // prepare buffer to save ips from all ranks
    char *all_str = NULL;
    int all_len = (max_len + 1) * g_world_size;
    all_str = (char *)malloc(all_len * sizeof(char));
    memset(all_str, '\0', all_len);

    // gather ips from all ranks
    MPI_Allgather(my_str_padded, max_len + 1, MPI_CHAR, all_str, max_len + 1, MPI_CHAR, MPI_COMM_WORLD);

    // restore all ips from char* to vector<string>
    char *str_idx = all_str;
    for (int rank_idx = 0; rank_idx < g_world_size; rank_idx++) {
        ips.push_back(str_idx);
        str_idx = str_idx + max_len + 1;
    }
}

// Parse raw output of ib command
float process_raw_output(string output) {
    float res = -1.0;
    try {
        vector<string> lines;
        boost::split(lines, output, boost::is_any_of("\n"), boost::token_compress_on);
        regex re("\\d+\\s+\\d+\\s+\\d+\\.\\d+\\s+(\\d+\\.\\d+)\\s+\\d+\\.\\d+");
        for (string line : lines) {
            smatch m;
            if (regex_search(line, m, re))
                res = std::max(res, stof(m.str(1)));
        }
    } catch (const std::exception &e) {
        std::cout << "Error: failed to parse raw_output: " << output << std::endl;
    }

    return res;
}

// Run ib command on server/client with server ip
float run_cmd(string cmd_prefix, int port, bool server, string ip) {
    // client sleep 1s in case that client starts before server
    if (!server) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    string command, output;
    try {
        // exec command in termimal
        command = cmd_prefix + " -p " + to_string(port);
        command = server ? command : command + " " + ip;
        output = exec(command.c_str());
    } catch (const std::exception &e) {
        std::cout << "Error: failed to exec command: " << command << std::endl;
    }

    // parse raw output and get result
    float result = process_raw_output(output);

    return result;
}

// The ranks in vector of (server, client) run commands parallel
vector<float> run_cmd_parallel(string cmd_prefix, const vector<std::pair<int, int>> &run_pairs_in_parallel,
                               const vector<int> &ports, const vector<string> &ips) {
    int size = run_pairs_in_parallel.size();
    // invoke function to run cmd in multi threads mode for each rank in the pairs
    unordered_map<int, std::future<float>> threads;
    int flag;
    for (int index = 0; index < size; index++) {
        int server_index = run_pairs_in_parallel[index].first;
        int client_index = run_pairs_in_parallel[index].second;
        if (server_index == g_world_rank) {
            flag = index;
            MPI_Send(&flag, 1, MPI_INT, client_index, index + 2 * size, MPI_COMM_WORLD);
            threads[2 * index] =
                (std::async(std::launch::async, run_cmd, cmd_prefix, ports[index], true, ips[server_index]));
        }
    }
    for (int index = 0; index < size; index++) {
        int server_index = run_pairs_in_parallel[index].first;
        int client_index = run_pairs_in_parallel[index].second;
        if (client_index == g_world_rank) {
            // in case that client starts before server
            MPI_Recv(&flag, 1, MPI_INT, server_index, index + 2 * size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            threads[2 * index + 1] =
                (std::async(std::launch::async, run_cmd, cmd_prefix, ports[index], false, ips[server_index]));
        }
    }
    // send the result of client to rank ROOT_RANK
    for (auto &thread : threads) {
        std::future_status status;
        float client_result = -1.0;
        status = thread.second.wait_for(std::chrono::seconds(10));
        if (status == std::future_status::timeout) {
            throw exception("thread timeout");
        } else if (status == std::future_status::ready) {
            client_result = thread.second.get();
        }
        int index = thread.first;
        if (index % 2 == 1) {
            MPI_Send(&client_result, 1, MPI_FLOAT, ROOT_RANK, index, MPI_COMM_WORLD);
        }
    }
    // rank ROOT_RANK recv results
    vector<float> results;
    if (g_world_rank == ROOT_RANK) {
        results.resize(size);
        for (int index = 0; index < size; index++) {
            int client_index = run_pairs_in_parallel[index].second;
            MPI_Recv(&results[index], 1, MPI_INT, client_index, index * 2 + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    return results;
}

vector<vector<float>> run_benchmark(const Args &args, vector<vector<std::pair<int, int>>> config,
                                    const vector<string> &ips) {
    vector<vector<float>> results;
    for (auto &line : config) {
        // Get ports for each run for single line of config
        auto ports = prepare_ports(line);
        // Insert barrier to sync before each run
        MPI_Barrier(MPI_COMM_WORLD);
        // run commands parallel for single line of config
        vector<float> results_single_line = run_cmd_parallel(args.cmd_prefix, line, ports, ips);
        // collect results for each run
        results.push_back(results_single_line);
        if (g_world_rank == ROOT_RANK) {
            std::cout << "results from rank ROOT_RANK: ";
            for (auto res : results_single_line) {
                std::cout << res << ",";
            }
            std::cout << endl;
        }
    }
    return results;
}

// Save results and output to csv file
void output_to_file(const std::string cmd_prefix, const vector<vector<std::pair<int, int>>> &config,
                    vector<vector<float>> &results, const std::string filename = "results.csv") {
    ofstream out(filename);
    if (!out) {
        throw exception("Error: failed to open output file.");
    }
    // output command predix
    out << "command prefix: " << cmd_prefix << std::endl;
    // output config file contents
    out << "config:" << std::endl;
    for (auto &line : config) {
        for (auto &pair : line) {
            out << pair.first << "," << pair.second << ";";
        }
        out << std::endl;
    }
    // output results
    out << "results:" << std::endl;
    for (auto &line : results) {
        for (int i = 0; i < line.size(); i++) {
            out << line[i] << ",";
        }
        out << std::endl;
    }
}

int main(int argc, char **argv) {
    try {
        // Initialize the MPI environment.
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

        // Get the number of ranks
        MPI_Comm_size(MPI_COMM_WORLD, &g_world_size);

        // Get the rank of the process
        MPI_Comm_rank(MPI_COMM_WORLD, &g_world_rank);

        // Get the name of the processor
        int name_len;
        MPI_Get_processor_name(g_processor_name, &name_len);

        // Get and parse command line arguments
        Args args;
        load_args(argc, argv, args);

        // Load and parse running config from file
        vector<vector<std::pair<int, int>>> config = load_config(args.input_config);

        // Get ips of all ranks
        vector<string> ips;
        gather_ips(ips);

        // Run validation benchmark
        vector<vector<float>> results = run_benchmark(args, config, ips);

        // rank ROOT_RANK output the results to file
        if (g_world_rank == ROOT_RANK) {
            output_to_file(args.cmd_prefix, config, results, args.output_path);
        }

        // Finalize the MPI environment. No more MPI calls can be made after this
        MPI_Finalize();
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const std::runtime_error &e) {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
