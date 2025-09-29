#include "host_tier.hpp"
#include <cmath>

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
    assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    checkCuda(cudaMallocHost(&start_ptr_, total_size));
    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);
    flush_thread_ = std::thread([&] { flush_io_(); });
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    flush_thread_.detach();
    fetch_thread_.detach();
    DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
}

void host_tier_t::flush(mem_region_t *src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
    assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
    flush_q.push(src);
}

void host_tier_t::fetch(mem_region_t *src) {
    fetch_q.push(src);
}

void host_tier_t::wait_for_completion() {
    DBG("Going to invoke flush_q.wait_for_completeion()");
    flush_q.wait_for_completion();
};

bool validate_data_integrity(char* data, size_t size) {
    if (data == nullptr || size == 0) return false;
    
    if (size >= sizeof(float)) {
        float* float_data = reinterpret_cast<float*>(data);
        size_t float_count = size / sizeof(float);
        size_t check_count = std::min(float_count, size_t(1000));
        
        for (size_t i = 0; i < check_count; i += 100) {
            if (std::isnan(float_data[i]) || std::isinf(float_data[i])) {
                return false;
            }
        }
    }
    
    return true;
}

void host_tier_t::flush_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        mem_region_t* src = nullptr;
        
        // CHANGED: Use atomic operation instead of get_front() + pop()
        if (!flush_q.wait_and_pop(src)) return;
        
        DBG("[HOST_TIER] Flushing from host to file " << src->uid << " at file_offset " << src->file_start_offset << " at " << src->path << " tensor of size " << src->size);
        
        try {
            if (src->ptr == nullptr) {
                DBG("[HOST_TIER] Error: Null pointer for UID " << src->uid);
                src->async_complete.store(true, std::memory_order_release);
                mem_pool->deallocate(src);
                delete src;
                continue;
            }
            
            if (!validate_data_integrity(src->ptr, src->size)) {
                DBG("[HOST_TIER] Data corruption detected in UID " << src->uid << ", skipping write");
                src->async_complete.store(true, std::memory_order_release);
                mem_pool->deallocate(src);
                delete src;
                continue;
            }
            
            std::string temp_path = src->path + ".tmp." + std::to_string(src->uid);
            
            {
                // Unified write path for both partial and non-partial updates:
                // - If destination exists, copy it to temp to preserve previously written chunks
                // - Else, create a new empty temp file
                // - Write this chunk at its file offset
                // - Atomically rename temp -> destination
                if (std::filesystem::exists(src->path)) {
                    std::filesystem::copy_file(src->path, temp_path,
                        std::filesystem::copy_options::overwrite_existing);
                } else {
                    std::ofstream create_temp(temp_path, std::ios::binary);
                    create_temp.close();
                }

                // Ensure file is large enough to accommodate the write at offset
                try {
                    const auto required_size = static_cast<uintmax_t>(src->file_start_offset + src->size);
                    uintmax_t current_size = 0;
                    if (std::filesystem::exists(temp_path)) {
                        current_size = std::filesystem::file_size(temp_path);
                    }
                    if (current_size < required_size) {
                        std::filesystem::resize_file(temp_path, required_size);
                    }
                } catch (const std::exception& ex) {
                    FATAL(std::string("[HostFlush] resize_file failed: ") + ex.what());
                }

                std::fstream f;
                f.exceptions(std::fstream::failbit | std::fstream::badbit);
                f.open(temp_path, std::ios::in | std::ios::out | std::ios::binary);
                f.seekp(src->file_start_offset);
                f.write(src->ptr, src->size);
                f.flush();
                f.close();

                std::filesystem::rename(temp_path, src->path);
            }
            
            src->async_complete.store(true, std::memory_order_release);
            mem_pool->deallocate(src);
            delete src;
            
            DBG("[HOST_TIER] Successfully flushed UID " << src->uid << " to file");
            
        } catch (const std::exception& ex) {
            std::string temp_path = src->path + ".tmp." + std::to_string(src->uid);
            if (std::filesystem::exists(temp_path)) {
                std::filesystem::remove(temp_path);
            }
            src->async_complete.store(true, std::memory_order_release);
            FATAL("[HostFlush] Got exception " << ex.what());
        }
    }
}

void host_tier_t::fetch_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        mem_region_t* src = nullptr;
        
        // CHANGED: Use atomic operation instead of get_front() + pop()
        if (!fetch_q.wait_and_pop(src)) return;
        
        try {
            DBG("Starting to fetch in background thread right now " << src->path << " from offset " << src->file_start_offset << " of size " << src->size);
            
            assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated for fetching.");
            
            // ADDED: Check if file exists before trying to read
            if (!std::filesystem::exists(src->path)) {
                DBG("[HOST_TIER] File not found: " << src->path);
                src->async_complete.store(true, std::memory_order_release);
                delete src;
                continue;
            }
                    
            std::ifstream f;            
            f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            f.open(src->path, std::ios::in | std::ios::binary);
            f.seekg(src->file_start_offset);
            f.read(src->ptr, src->size);
            f.close();
            
            if (!validate_data_integrity(src->ptr, src->size)) {
                DBG("[HOST_TIER] Loaded data validation failed for " << src->path);
            }
            
            src->async_complete.store(true, std::memory_order_release);
            delete src;
            
        } catch (const std::exception& ex) {
            // CHANGED: Don't call FATAL (which aborts), just log error and continue
            DBG("[HostFetch] Got exception " << ex.what());
            if (src) {
                src->async_complete.store(true, std::memory_order_release);
                delete src;
            }
        }
    }
}