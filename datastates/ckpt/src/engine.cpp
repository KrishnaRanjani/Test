#include "engine.hpp"
#include <filesystem>

datastates_llm_t::datastates_llm_t(size_t host_cache_size, int gpu_id_, int rank_): 
    gpu_id(gpu_id_), rank(rank_), warmup_steps(100), current_epoch(0), warmup_complete(false) {
    try {
        DBG("DataStates initing: GPU: " << gpu_id << ", host cache (MB): " << (host_cache_size >> 20) << ", warmup steps: " << warmup_steps);
        checkCuda(cudaSetDevice(gpu_id));
        is_active = true;
        int num_threads = 1;
        size_t gpu_cache = 0;
        host_tier = new host_tier_t(gpu_id, num_threads, host_cache_size);
        gpu_tier = new gpu_tier_t(gpu_id, num_threads, gpu_cache);
        gpu_tier->set_successor_tier(host_tier);
    } catch(std::exception& e) {
        FATAL("Standard exception caught in datastates init: " << e.what());
    }
}

// FIXED: Better tensor validation and error recovery
void datastates_llm_t::ckpt_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path) {
    try {
        if (skip_checkpoint_this_step.load()) {
            total_checkpoints_skipped++;
            DBG("Skipping tensor checkpoint due to NaN detection (total skipped: " << total_checkpoints_skipped.load() << ")");
            return;
        }
        
        uint64_t uid = local_uid++;
        DBG("Going to checkpoint tensor of UID " << uid << " and size " << size << " at offset " << file_offset);
        
        bool is_partial = std::filesystem::exists(path);
        
        torch::Tensor safe_tensor;
        if (torch::isnan(t).any().item<bool>()) {
            DBG("NaN detected in tensor UID " << uid << ", skipping");
            total_checkpoints_skipped++;
            set_skip_checkpoint();
            return;
        }
        
        if (!t.is_contiguous()) {
            safe_tensor = t.detach().contiguous().clone();
        } else {
            safe_tensor = t.detach().clone();
        }
        
        if (safe_tensor.device().is_cuda()) {
            assert((safe_tensor.device().is_cuda() && safe_tensor.device().index() == gpu_id) && "Tensor not on the same GPU as ckpt engine");
            mem_region_t* m = new mem_region_t(version, uid, static_cast<char *>(safe_tensor.data_ptr()), size, file_offset, path, GPU_TIER, is_partial);
            
            m->tensor_ref = safe_tensor;
            gpu_tier->flush(m);
            return;
        } 
        
        mem_region_t* m = new mem_region_t(version, uid, static_cast<char *>(safe_tensor.data_ptr()), size, file_offset, path, HOST_PINNED_TIER, is_partial);
        m->tensor_ref = safe_tensor;
        host_tier->flush(m);
        return;
    } catch (std::exception &e) {
        DBG("Exception caught in ckpt_tensor: " << e.what());
        return;
    }
}

void datastates_llm_t::restore_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path) {
    try {
        if (t.device().is_cuda()) 
            FATAL("Restoring GPU tensor is not yet supported");
        uint64_t uid = local_uid++;
        DBG("Going to restore from " << path << " tensor of size " << size << " at file offset " << file_offset);
        mem_region_t* m = new mem_region_t(version, uid, static_cast<char *>(t.data_ptr()), size, file_offset, path, HOST_PINNED_TIER);
        m->tensor_ref = t;
        host_tier->fetch(m);
        return;
    } catch (std::exception &e) {
        DBG("Exception caught in restore_tensor: " << e.what());
        return;
    }
}

void datastates_llm_t::wait() {
    try {
        gpu_tier->wait_for_completion();
    } catch (std::exception &e) {
        DBG("Exception caught in wait: " << e.what());
    }
}

void datastates_llm_t::shutdown() {
    try {
        delete gpu_tier;
        delete host_tier;
        return;
    } catch (std::exception &e) {
        DBG("Exception caught in shutdown: " << e.what());
    }
}

bool datastates_llm_t::should_force_full_checkpoint() {
    return current_epoch < warmup_steps;
}

void datastates_llm_t::increment_checkpoint_epoch() {
    current_epoch++;
    if (current_epoch >= warmup_steps && !warmup_complete) {
        warmup_complete = true;
        DBG("Warmup phase completed at epoch " << current_epoch);
    }
}

bool datastates_llm_t::is_warmup_complete_status() {
    return warmup_complete;
}

int datastates_llm_t::get_current_epoch() {
    return current_epoch;
}

void datastates_llm_t::set_skip_checkpoint() {
    skip_checkpoint_this_step = true;
}

void datastates_llm_t::reset_checkpoint_skip() {
    skip_checkpoint_this_step = false;
}

int datastates_llm_t::get_skipped_checkpoint_count() {
    return total_checkpoints_skipped.load();
}