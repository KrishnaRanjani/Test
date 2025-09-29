#ifndef __DATASTATES_MEM_POOL_HPP
#define __DATASTATES_MEM_POOL_HPP

#include "mem_region.hpp"
#include "utils.hpp"
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>

class mem_pool_t {
    char* start_ptr_ = nullptr;
    size_t total_size_ = 0;
    size_t curr_size_ = 0;
    size_t head_ = 0;
    size_t tail_ = 0;
    int rank_ = -1;
    cudaMemoryType device_type_;
    std::deque<mem_region_t*> mem_q_;
    std::unordered_map<uint64_t, size_t> alloc_map_;
    std::mutex mem_mutex_;
    std::condition_variable mem_cv_;
    bool is_active = true;
    
public:
    mem_pool_t(char* start_ptr, size_t total_size, int rank);
    ~mem_pool_t();
    
    size_t get_free_size();
    size_t get_capacity();
    void allocate(mem_region_t* m);
    void deallocate(mem_region_t* m);
    
private:
    void assign_(mem_region_t* m);
    void print_trace_();
};

#endif //__DATASTATES_MEM_POOL_HPP