#ifndef __DATASTATES_ATOMIC_QUEUE_HPP
#define __DATASTATES_ATOMIC_QUEUE_HPP

#include "mem_region.hpp"
#include "defs.hpp"
#include "utils.hpp"
#include <mutex>
#include <atomic>
#include <condition_variable>

class atomic_queue_t {
    std::deque<mem_region_t*> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> is_active = true;
public:
    atomic_queue_t() {};
    ~atomic_queue_t() {};
    
    void push(mem_region_t* src) {
        std::unique_lock<std::mutex> lck(mtx);
        q.push_back(src);
        lck.unlock();
        cv.notify_all();
    };
    
    // Add missing get_front() method called in gpu_tier.cpp and host_tier.cpp
    mem_region_t* get_front() {
        std::unique_lock<std::mutex> lck(mtx);
        if (q.empty()) {
            return nullptr;
        }
        return q.front();
    };
    
    //  Add missing pop() method called in gpu_tier.cpp and host_tier.cpp
    void pop() {
        std::unique_lock<std::mutex> lck(mtx);
        if (!q.empty()) {
            q.pop_front();
        }
        lck.unlock();
        cv.notify_all();
    };
    
    // ADDED: Replace get_front() + pop() with atomic operation
    bool wait_and_pop(mem_region_t*& result) {
        std::unique_lock<std::mutex> lck(mtx);
        while(q.empty() && is_active) {
            cv.wait(lck);
        }
        
        if (!is_active || q.empty()) {
            return false;
        }
        
        result = q.front();
        q.pop_front();
        lck.unlock();
        cv.notify_all();
        return true;
    };
    
    void wait_for_completion() {
        std::unique_lock<std::mutex> lck(mtx);
        while(q.size() > 0 && is_active)
            cv.wait(lck);
        lck.unlock();
        cv.notify_all();
    }
    
    void set_inactive() {
        std::unique_lock<std::mutex> lck(mtx);
        is_active = false;
        lck.unlock();
        cv.notify_all();
    };
    
    bool wait_for_item() {
        std::unique_lock<std::mutex> lck(mtx);
        while(q.empty() && is_active)
            cv.wait(lck);
        lck.unlock();
        cv.notify_all();
        return is_active;
    };
    
    //  Add size() method for debugging queue state
    size_t size() {
        std::unique_lock<std::mutex> lck(mtx);
        return q.size();
    };
    
    //  Add empty() check method
    bool empty() {
        std::unique_lock<std::mutex> lck(mtx);
        return q.empty();
    };
};

#endif //__DATASTATES_ATOMIC_QUEUE_HPP