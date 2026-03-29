#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// ThreadPool
//
// Work-stealing not needed for our tile-parallel workload — tiles are
// independent and uniform in cost. A simple lock-free atomic counter over
// a pre-filled task queue is sufficient and avoids mutex contention on the
// hot path.
//
// Usage:
//   ThreadPool pool;                    // hardware_concurrency threads
//   pool.parallelFor(numTiles, [&](uint32_t tileIdx) {
//       renderTile(tiles[tileIdx]);
//   });
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(uint32_t numThreads = 0) {
        uint32_t n = numThreads > 0
            ? numThreads
            : std::max(1u, std::thread::hardware_concurrency());

        m_threads.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            m_threads.emplace_back([this] { workerLoop(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock lock(m_mutex);
            m_shutdown = true;
        }
        m_cv.notify_all();
        for (auto& t : m_threads) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    uint32_t numThreads() const {
        return static_cast<uint32_t>(m_threads.size());
    }

    // Run `func(i)` for i in [0, count) across the thread pool.
    // Blocks until all iterations are complete.
    void parallelFor(uint32_t count, std::function<void(uint32_t)> func) {
        if (count == 0) return;

        // Atomic counter: each thread claims the next unclaimed index
        std::atomic<uint32_t> next{0};
        std::atomic<uint32_t> done{0};
        std::mutex             doneMutex;
        std::condition_variable doneCv;

        auto task = [&] {
            while (true) {
                uint32_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) break;
                func(i);
            }
            uint32_t completed = done.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (completed == std::min(count, numThreads())) {
                std::unique_lock lock(doneMutex);
                doneCv.notify_one();
            }
        };

        // Enqueue one task per thread (each task internally loops via `next`)
        uint32_t numTasks = std::min(count, numThreads());
        {
            std::unique_lock lock(m_mutex);
            for (uint32_t i = 0; i < numTasks; ++i)
                m_queue.push(task);
        }
        m_cv.notify_all();

        // Wait for all tasks to complete
        std::unique_lock lock(doneMutex);
        doneCv.wait(lock, [&] {
            return done.load(std::memory_order_acquire) >= numTasks;
        });
    }

    // Submit a single task asynchronously (fire-and-forget)
    void submit(std::function<void()> task) {
        {
            std::unique_lock lock(m_mutex);
            m_queue.push(std::move(task));
        }
        m_cv.notify_one();
    }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(m_mutex);
                m_cv.wait(lock, [this] {
                    return m_shutdown || !m_queue.empty();
                });
                if (m_shutdown && m_queue.empty()) return;
                task = std::move(m_queue.front());
                m_queue.pop();
            }
            task();
        }
    }

    std::vector<std::thread>          m_threads;
    std::queue<std::function<void()>> m_queue;
    std::mutex                        m_mutex;
    std::condition_variable           m_cv;
    bool                              m_shutdown = false;
};

} // namespace anacapa
