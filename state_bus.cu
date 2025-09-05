#include "state_bus.hpp"


// Example planner-facing publisher (toy). In a real system this would be an LCM/ROS2 pub
// and would perform time alignment: t_cam + t_offset -> t_body.
extern "C" void step5_publish_example(StateRing<256>& ring, TailLatencyGuard& g, double t_now,
                                      const float* T_wb_pos, const float* T_wb_quat, const float* vel,
                                      const float* cov_pos_diag, uint32_t seq, double last_frame_ms){
    StateEstimate s{};
    s.t_sec=t_now;
    std::memcpy(s.pos,T_wb_pos,3*sizeof(float));
    std::memcpy(s.quat,T_wb_quat,4*sizeof(float));
    
    std::memcpy(s.vel,vel,3*sizeof(float));
    std::memcpy(s.cov_pos,cov_pos_diag,3*sizeof(float));
    s.seq=seq;
    
    bool ok = ring.push(s);
    if (!ok) {
        // backpressure: drop oldest by popping once, then push
        StateEstimate tmp; ring.pop(tmp); ring.push(s);
    }
    g.add(last_frame_ms); g.tick();
    if (g.degrade){
        // Signal upstream front-end to reduce load (e.g., higher FAST threshold or fewer features)
        // (left as a shared atomic flag in a real system)
        fprintf(stderr, "[Step5] Degrade mode ON (p95=%.2fms) -> raise FAST thresh / skip descriptors)\n", g.percentile95());
    }
}
