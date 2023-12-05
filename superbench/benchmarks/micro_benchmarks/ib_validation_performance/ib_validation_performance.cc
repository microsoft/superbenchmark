// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// IB validation tool is a tool to validate IB traffic of different pattern in multi nodes flexibly
// input:
//  cmd_prefix: the prefix of command to run
//  input_config: the path of input config file, the format of config file is as the following,
//    each row will run in parallel, different rows will run in sequence, each pair is (server index, client index)
//    1,0;2,0;3,0
//    0,1;2,1;3,1
//    0,2;1,2;3,2
//    0,3;1,3;2,3
//  output_path: the path of output csv file
//  hostfile: the path of hostfile, containing the hostname of all ranks in rank order,
//    each line is the hostname for the rank

#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <regex>
#include <stdexcept>
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
#define SERVER_MAX_THREADS 25000
#define MAX_THREADS 65535

int g_world_size;
int g_world_rank;
char g_processor_name[MPI_MAX_PROCESSOR_NAME];

int local_size;

// The struct to store command line arguments
struct Args {
    // Timeout for each command
    int timeout;
    // The prefix of command to run
    std::string send_cmd_prefix;
    std::string recv_cmd_prefix;
    // The path of input config file
    std::string input_config;
    // The path of output csv file
    std::string output_path;
    // The path of hostfile
    std::string hostfile;
};

// Parse and save command line arguments
void load_args(int argc, char *argv[], Args &args) {
    // Get and parse command line arguments
    boost::program_options::options_description opt("all options");
    opt.add_options()("timeout,t", boost::program_options::value<int>(&args.timeout)->default_value(120),
                      "timeout of each command")("send_cmd_prefix,c",
                                                 boost::program_options::value<std::string>(&args.send_cmd_prefix)
                                                     ->default_value("ib_write_bw -s 33554432 -d ib0"),
                                                 "ib command prefix")(
        "recv_cmd_prefix,c",
        boost::program_options::value<std::string>(&args.recv_cmd_prefix)
            ->default_value("ib_write_bw -s 33554432 -d ib0"),
        "ib command prefix")(
        "input_config,i", boost::program_options::value<std::string>(&args.input_config)->default_value("config.txt"),
        "the path of input config file")(
        "hostfile,h", boost::program_options::value<std::string>(&args.hostfile)->default_value("hostfile"),
        "the path of hostfile")(
        "output_path,o", boost::program_options::value<std::string>(&args.output_path)->default_value("output.csv"),
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
        std::cout << "Timeout for each command is: " << args.timeout << std::endl;
        std::cout << "The prefix of cmd to run is: " << args.send_cmd_prefix << args.recv_cmd_prefix << std::endl;
        std::cout << "Load the config file from: " << args.input_config << std::endl;
        std::cout << "Output will be saved to: " << args.output_path << std::endl;
    }
}

// Load and parse config file to vector
vector<vector<std::pair<int, int>>> load_config(string filename = "config.txt") {
    // read contents from file
    vector<std::string> config;
    ifstream in(filename);
    string line;

    if (in) {
        while (getline(in, line)) {
            if (line.size() > 0)
                config.push_back(line);
        }
    } else {
        throw std::runtime_error("Error: Failed to open config file.");
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
            vector<int> s_occurrence(g_world_size / local_size, 0), occurrence(g_world_size / local_size, 0);
            for (const auto &pair : run_in_parallel) {
                // split pair by ","
                size_t quote = pair.find(',');
                if (quote == pair.npos) {
                    throw std::runtime_error("Error: Invalid config format.");
                }
                int first = stoi(pair.substr(0, quote));
                int second = stoi(pair.substr(quote + 1));

                occurrence[first]++;
                occurrence[second]++;
                s_occurrence[first]++;
                // limit the maximum threads of each node no more than 65535 and server threads no more than 25000 at
                // the same time because by default a node can use (32768-60999) ports
                if (s_occurrence[first] * local_size >= SERVER_MAX_THREADS ||
                    occurrence[second] * local_size >= MAX_THREADS || occurrence[first] * local_size >= MAX_THREADS) {
                    if (g_world_rank == ROOT_RANK)
                        std::cout << "Warning: split the line due to the limit of maximum threads nums" << std::endl;
                    run_in_total.emplace_back(run_pairs_in_parallel);
                    run_pairs_in_parallel.clear();
                    occurrence.assign(g_world_size / local_size, 0);
                    s_occurrence.assign(g_world_size / local_size, 0);
                }
                run_pairs_in_parallel.emplace_back(first, second);
            }
            run_in_total.emplace_back(run_pairs_in_parallel);
        }
    } catch (...) {
        std::throw_with_nested(std::runtime_error("Error: Invalid config format."));
    }
    if (g_world_rank == ROOT_RANK) {
        std::cout << "config: " << std::endl;
        for (const vector<std::pair<int, int>> &line : run_in_total) {
            for (const std::pair<int, int> &pair : line) {
                std::cout << pair.first << "," << pair.second << ";";
            }
            std::cout << endl;
        }
        std::cout << "config end" << std::endl;
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
        throw std::runtime_error("Error: popen() failed!");
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

// Use socket to get a free port
int get_available_port() {
    struct sockaddr_in server_addr;
    int sockfd;
    // open a socket
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        throw std::runtime_error("Error: failed to open socket.");
    }
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = 0;
    inet_aton("127.0.0.1", &server_addr.sin_addr);
    bzero((char *)&server_addr, sizeof(server_addr));
    // assign a port number
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        throw std::runtime_error("Error: failed to bind socker.");
    }
    socklen_t len_inet = sizeof(server_addr);
    // get socket name
    if (getsockname(sockfd, (struct sockaddr *)&server_addr, &len_inet) < 0) {
        throw std::runtime_error("Error: failed to get socketname.");
    }
    // get the port number
    int local_port = ntohs(server_addr.sin_port);
    // make the port can be reused
    int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
        throw std::runtime_error("Error: failed to set socket.");
    }
    return local_port;
}

// Get and broadcast ports for each run
vector<int> prepare_ports(const vector<std::pair<int, int>> &run_pairs_in_parallel) {
    vector<int> ports(run_pairs_in_parallel.size() * local_size);

    for (size_t index = 0; index < run_pairs_in_parallel.size(); index++) {
        int server_index = run_pairs_in_parallel[index].first;
        for (int rank = 0; rank < local_size; rank++) {
            // server get a free port and send to rank ROOT_RANK
            if (server_index * local_size + rank == g_world_rank) {
                int port = get_available_port();
                MPI_Send(&port, 1, MPI_INT, ROOT_RANK, index * local_size + rank, MPI_COMM_WORLD);
            }
            // rank ROOT_RANK recv port from server
            if (g_world_rank == ROOT_RANK) {
                int port;
                MPI_Recv(&port, 1, MPI_INT, server_index * local_size + rank, index * local_size + rank, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                ports[index * local_size + rank] = port;
            }
        }
    }
    // rank ROOT_RANK broadcast ports to all ranks
    MPI_Bcast(ports.data(), ports.size(), MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    return ports;
}

// Get hostnames for all ranks
void gather_hostnames(vector<string> &hostnames, string filename) {
    ifstream in(filename);
    string line;

    if (in) {
        while (getline(in, line)) {
            if (line.size() > 0)
                hostnames.push_back(line);
        }
    } else {
        throw std::runtime_error("Error: Failed to open hostfile.");
    }
    if (int(hostnames.size()) < g_world_size / local_size) {
        throw std::runtime_error("Error: Invalid hostfile.");
    }
}

// Parse raw output of ib command
// Sample of ib bw command raw
// #bytes     #iterations BW peak[Gb/sec]    BW average[Gb/sec]  MsgRate[Mpps]
// 8388608    5000            196.08             195.76            0.002917
// Sample of ib latency command raw output
// #bytes  #iterations    t_min    t_max  t_typical   t_avg    t_stdev  99% percentile   99.9% percentile
// 8388608    5000        581.27   876.26   594.87    595.50     3.33       601.65          621.14
// parsed result:
// 195.76 (BW average)
// 595.50 (t_avg)
float process_raw_output(string output) {
    float res = -1.0;
    try {
        string pattern;
        vector<string> lines;
        boost::split(lines, output, boost::is_any_of("\n"), boost::token_compress_on);
        if (output.find("BW") != string::npos) {
            pattern = "\\d+\\s+\\d+\\s+\\d+\\.\\d+\\s+(\\d+\\.\\d+)\\s+\\d+\\.\\d+";
        } else {
            pattern = "\\d+\\s+\\d+\\s+\\d+\\.\\d+\\s+\\d+\\.\\d+\\s+\\d+\\.\\d+"
                      "\\s+(\\d+\\.\\d+)\\s+\\d+\\.\\d+\\s+\\d+\\.\\d+\\s+\\d+\\.\\d+";
        }
        regex re(pattern);
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

// Run ib command on server/client with server hostname
float run_cmd(string cmd_prefix, int timeout, int port, bool server, string hostname) {
    // client sleep 1s in case that client starts before server
    if (!server) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    string command, output;
    try {
        // exec command in termimal
        command = "timeout " + to_string(timeout) + " " + cmd_prefix + " -p " + to_string(port);
        command = server ? command : command + " " + hostname;
        output = exec(command.c_str());
    } catch (const std::exception &e) {
        std::cout << "Error: failed to exec command: " << command << ",msg: " << e.what() << std::endl;
    }

    // parse raw output and get result
    float result = process_raw_output(output);

    return result;
}

// The ranks in vector of (server, client) run commands parallel
vector<float> run_cmd_parallel(string send_cmd_prefix, string recv_cmd_prefix, int timeout,
                               const vector<std::pair<int, int>> &run_pairs_in_parallel, const vector<int> &ports,
                               const vector<string> &hostnames) {
    // invoke function to run cmd in multi threads mode for each rank in the pairs
    unordered_map<int, std::future<float>> threads;
    int flag;
    for (size_t index = 0; index < run_pairs_in_parallel.size(); index++) {
        for (int rank = 0; rank < local_size; rank++) {
            int rank_index = index * local_size + rank,
                server_index = run_pairs_in_parallel[index].first * local_size + rank,
                client_index = run_pairs_in_parallel[index].second * local_size + rank;
            if (server_index == g_world_rank) {
                flag = index;
                MPI_Send(&flag, 1, MPI_INT, client_index, rank_index, MPI_COMM_WORLD);
                threads[2 * rank_index] = (std::async(std::launch::async, run_cmd, recv_cmd_prefix, timeout,
                                                      ports[rank_index], true, hostnames[server_index / local_size]));
            }
            if (client_index == g_world_rank) {
                // in case that client starts before server
                MPI_Recv(&flag, 1, MPI_INT, server_index, rank_index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                threads[2 * rank_index + 1] =
                    (std::async(std::launch::async, run_cmd, send_cmd_prefix, timeout, ports[rank_index], false,
                                hostnames[server_index / local_size]));
            }
        }
    }

    // send the result of client to rank ROOT_RANK
    for (auto &thread : threads) {
        std::future_status status;
        float client_result = -1.0;
        status = thread.second.wait_for(std::chrono::seconds(timeout));
        if (status == std::future_status::timeout) {
            std::cout << "Error: thread timeout" << std::endl;
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
        results.resize(run_pairs_in_parallel.size() * local_size);
        for (size_t index = 0; index < run_pairs_in_parallel.size(); index++) {
            for (int rank = 0; rank < local_size; rank++) {
                int rank_index = index * local_size + rank,
                    client_index = run_pairs_in_parallel[index].second * local_size + rank;
                MPI_Recv(&results[rank_index], 1, MPI_FLOAT, client_index, 2 * rank_index + 1, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
    }
    return results;
}

vector<vector<float>> run_benchmark(const Args &args, vector<vector<std::pair<int, int>>> config,
                                    const vector<string> &hostnames) {
    vector<vector<float>> results;
    for (auto &line : config) {
        // Get ports for each run for single line of config
        auto ports = prepare_ports(line);
        // Insert barrier to sync before each run
        MPI_Barrier(MPI_COMM_WORLD);
        // run commands parallel for single line of config
        vector<float> results_single_line =
            run_cmd_parallel(args.send_cmd_prefix, args.recv_cmd_prefix, args.timeout, line, ports, hostnames);
        // collect results for each run
        results.push_back(results_single_line);
    }
    // output the results to stdout of ROOT_RANK by default
    if (g_world_rank == ROOT_RANK) {
        std::cout << "results from rank ROOT_RANK: " << std::endl;
        for (vector<float> line : results) {
            for (size_t i = 0; i < line.size(); i++) {
                std::cout << line[i] << ((i + 1) % local_size ? " " : ",");
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
        throw std::runtime_error("Error: failed to open output file.");
    }
    // output command prefix
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
        for (size_t i = 0; i < line.size(); i++) {
            out << line[i] << ((i + 1) % local_size ? " " : ",");
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

        // Handle local size and rank
#if defined(OPEN_MPI)
        local_size = atoi(getenv("OMPI_COMM_WORLD_LOCAL_SIZE"));
        boost::replace_all(args.send_cmd_prefix, "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK");
        boost::replace_all(args.recv_cmd_prefix, "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK");
#elif defined(MPICH)
        local_size = atoi(getenv("MPI_LOCALNRANKS"));
        boost::replace_all(args.send_cmd_prefix, "LOCAL_RANK", "MPI_LOCALRANKID");
        boost::replace_all(args.recv_cmd_prefix, "LOCAL_RANK", "MPI_LOCALRANKID");
#else
        local_size = atoi(getenv("LOCAL_SIZE"));
        std::cout << "Warning: unknown mpi used." << std::endl;
#endif

        // Load and parse running config from file
        vector<vector<std::pair<int, int>>> config = load_config(args.input_config);

        // Get hostnames of all ranks
        vector<string> hostnames;
        gather_hostnames(hostnames, args.hostfile);

        // Run validation benchmark
        vector<vector<float>> results = run_benchmark(args, config, hostnames);

        // rank ROOT_RANK output the results to file
        if (g_world_rank == ROOT_RANK) {
            if (args.output_path.size() != 0)
                output_to_file(args.send_cmd_prefix, config, results, args.output_path);
        }

        // Finalize the MPI environment. No more MPI calls can be made after this
        MPI_Finalize();
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
