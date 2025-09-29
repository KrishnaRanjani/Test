#ifndef __DATASTATES_MEM_REGION_HPP
#define __DATASTATES_MEM_REGION_HPP
#include <iostream>
#include <limits.h>
#include <deque>
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <torch/torch.h>
#include "defs.hpp"

struct mem_region_t {
    const int           version;
    const uint64_t      uid;
    char*               ptr;
    const size_t        size;
    const size_t        file_start_offset;
    const std::string   path;
    TIER_TYPES          curr_tier_type;
    bool                is_partial_update = false;
    
    torch::Tensor       tensor_ref;
    std::atomic<bool>   async_complete{false};
    std::shared_ptr<char[]> checkpoint_buffer;
    
    mem_region_t(const int version_, const uint64_t uid_, char* const ptr_, 
        const size_t size_, const size_t file_start_offset_, const std::string path_, 
        TIER_TYPES tier, bool partial = false): 
        version(version_), uid(uid_), ptr(ptr_), size(size_), 
        file_start_offset(file_start_offset_), path(path_), 
        curr_tier_type(tier), is_partial_update(partial) {};
    
    mem_region_t(const mem_region_t* other, TIER_TYPES next_tier): 
        version(other->version), uid(other->uid), ptr(nullptr), size(other->size), 
        file_start_offset(other->file_start_offset), path(other->path), 
        curr_tier_type(next_tier), is_partial_update(other->is_partial_update),
        tensor_ref(other->tensor_ref), checkpoint_buffer(other->checkpoint_buffer) {};
    
    ~mem_region_t() {
        int wait_count = 0;
        while (!async_complete.load(std::memory_order_acquire) && wait_count < 10000) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            wait_count++;
        }
        
        if (wait_count >= 10000) {
            std::cerr << "Async completion timeout for UID " << uid << std::endl;
        }
    }
    
    mem_region_t& operator=(const mem_region_t&) = delete;
};

#endif //__DATASTATES_MEM_REGION_HPP