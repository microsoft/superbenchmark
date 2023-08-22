// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// ThreadPool is a simple thread pool implementation that supports enqueueing the task with the index of thread to use
// and custom arguments like task(thread_index, *args).
class ThreadPool {
  public:
    /**
     * @brief Construct a new ThreadPool object with the given number of threads.
     */
    ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back(&ThreadPool::worker, this, i);
        }
    }
    /**
     * @brief Destroy the ThreadPool object and join all threads.
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();

        for (auto &thread : threads) {
            thread.join();
        }
    }
    /**
     * @brief TaskWrapper is a wrapper of the task with the index of thread to use and custom arguments like
     * task(thread_index, *args).
     */
    template <typename R, typename F, typename... Args> struct TaskWrapper {
        std::shared_ptr<std::packaged_task<R(size_t)>> task;

        template <typename Callable, typename... CallableArgs> TaskWrapper(Callable &&f, CallableArgs &&...args) {
            task = std::make_shared<std::packaged_task<R(size_t)>>(
                [f, args...](size_t threadIdx) mutable { return f(threadIdx, args...); });
        }

        void operator()(size_t threadIdx) { (*task)(threadIdx); }
    };
    /**
     * @brief Enqueue enqueues the task with custom arguments and return the results of task when finished.
     */
    template <typename F, typename... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(size_t, Args...)>::type> {
        using ReturnType = typename std::result_of<F(size_t, Args...)>::type;

        TaskWrapper<ReturnType, F, Args...> wrapper(std::forward<F>(f), std::forward<Args>(args)...);
        std::future<ReturnType> res = wrapper.task->get_future();

        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.emplace(std::move(wrapper));
        }
        cv.notify_one();

        return res;
    }

  private:
    /**
     * @brief The worker function that dequeues the task and executes it for each thread index.
     */
    void worker(size_t threadIdx) {
        while (true) {
            std::function<void(size_t)> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return stop || !tasks.empty(); });

                if (stop && tasks.empty()) {
                    return;
                }

                task = tasks.front();
                tasks.pop();
            }

            task(threadIdx);
        }
    }

    std::vector<std::thread> threads;
    std::queue<std::function<void(size_t)>> tasks;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;
};
