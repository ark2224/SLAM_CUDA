#pragma once
#include <vector>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <cstdint>

struct StateEstimate {
    double t_sec;
    float  pos[3];
    float  quat[4];   // w,x,y,z
    float  vel[3];
    float  cov_pos[3];
    uint32_t seq;
};

template<int CAP>
struct StateRing {
    StateEstimate              buf[CAP];
    std::atomic<uint32_t>     head{0};
    std::atomic<uint32_t>     tail{0};
    bool push(const StateEstimate& s) {
        uint32_t h = head.load(std::memory_order_relaxed);
        uint32_t t = tail.load(std::memory_order_acquire);
        if (((h+1) % CAP) == t) return false;
        buf[h] = s;
        head.store((h+1) % CAP, std::memory_order_release);
        return true;
    }
    bool pop(StateEstimate& out) {
        uint32_t t = tail.load(std::memory_order_relaxed);
        uint32_t h = head.load(std::memory_order_acquire);
        if (t == h) return false;
        out = buf[t];
        tail.store((t+1) % CAP, std::memory_order_release);
        return true;
    }
};

struct TailLatencyGuard {
    std::vector<double> hist_ms;
    size_t cap;
    double p95_limit_ms;
    bool   degrade = false;

    TailLatencyGuard(size_t cap_, double p95_lim)
        : cap(cap_), p95_limit_ms(p95_lim) {}

    void add(double ms) {
        hist_ms.push_back(ms);
        if (hist_ms.size() > cap) hist_ms.erase(hist_ms.begin());
    }
    double percentile95() const {
        if (hist_ms.empty()) return 0.0;
        auto v = hist_ms;
        size_t k = (size_t)(0.95 * (v.size() - 1));
        std::nth_element(v.begin(), v.begin()+k, v.end());
        return v[k];
    }
    void tick() { degrade = (percentile95() > p95_limit_ms); }
};

extern "C" void step5_publish_example(StateRing<256>& ring, TailLatencyGuard& g, double t_now,
                                      const float* T_wb_pos, const float* T_wb_quat, const float* vel,
                                      const float* cov_pos_diag, uint32_t seq, double last_frame_ms);
