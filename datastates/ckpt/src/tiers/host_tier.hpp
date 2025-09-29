#ifndef __DATASTATES_HOST_TIER_HPP
#define __DATASTATES_HOST_TIER_HPP

#include "base_tier.hpp"
#include <fstream>
#include <filesystem>

class host_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
    std::atomic<bool> shutdown_requested{false};
public:
    host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    
    // FIXED: Proper shutdown sequence
    ~host_tier_t() {
        shutdown_requested.store(true);
        is_active = false;
        
        // Stop queues first
        flush_q.set_inactive();
        fetch_q.set_inactive();
        
        // Wait for current operations to complete
        wait_for_completion();
        
        // Give threads time to exit cleanly
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Clean up memory pool
        if (mem_pool) {
            delete mem_pool;
            mem_pool = nullptr;
        }
        
        // Free pinned memory
        if (start_ptr_) {
            cudaFreeHost(start_ptr_);
            start_ptr_ = nullptr;
        }
    };
    
    void flush(mem_region_t* m);
    void fetch(mem_region_t* m);
    void flush_io_();
    void fetch_io_();
    void wait_for_completion();
};

#endif // __DATASTATES_HOST_TIER_HPP