#ifndef __DATASTATES_HPP
#define __DATASTATES_HPP

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "tiers/host_tier.hpp"
#include "tiers/gpu_tier.hpp"

namespace py = pybind11;

static volatile uint64_t local_uid = 1;

class datastates_llm_t {
    host_tier_t* host_tier;
    gpu_tier_t* gpu_tier;
    bool is_active = true;
    int gpu_id = 0;
    int rank = -1;
    
    // Warmup tracking
    int warmup_steps;
    int current_epoch;
    bool warmup_complete;
    
    // NaN detection and checkpointing skip
    std::atomic<bool> skip_checkpoint_this_step{false};
    std::atomic<int> total_checkpoints_skipped{0};
    
public:
    datastates_llm_t(size_t host_cache_size, int gpu_id, int rank=-1);
    void ckpt_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    void restore_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    void wait();
    void shutdown();
    
    // Warmup methods
    bool should_force_full_checkpoint();
    void increment_checkpoint_epoch();
    bool is_warmup_complete_status();
    int get_current_epoch();
    
    // NaN checkpoint control methods
    void set_skip_checkpoint();
    void reset_checkpoint_skip();
    int get_skipped_checkpoint_count();
};

#endif // __DATASTATES_HPP