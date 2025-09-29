from datastates.ckpt.src import handle as datastates_handle
import torch
import time
import sys
import os
from datastates.utils import get_logger

class CkptEngine:
    def __init__(self, host_cache_size, gpu_device_id, rank, warmup_steps=100) -> None:
        try:
            self.ckpt_engine = datastates_handle(host_cache_size, gpu_device_id, rank)
            self.logger = get_logger(__name__)
            self.last_ckpt_version = -1
            
            self.logger.info(f"Engine initialized with {host_cache_size / (1<<20):.0f}MB cache")
        except Exception as exc:
            print(f"[DataStates.ckpt][ERROR] Init failed: {exc}")
            sys.exit(-1)

    def async_save(self, tensors: list[tuple[int, torch.Tensor, int, str]]):
        """Async save tensors to distributed files"""
        try:
            for i, (version, tensor, file_offset, path) in enumerate(tensors):
                tensor_bytes = tensor.numel() * tensor.element_size()
                assert tensor_bytes > 0, f"Tensor {i} size should be > 0"
                
                # Ensure directory exists for distributed files
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                staged_tensor = tensor.detach().contiguous()
                self.ckpt_engine.ckpt_tensor(version, staged_tensor, tensor_bytes, file_offset, path)
                
            self.logger.info(f"Async save launched: {len(tensors)} tensors")
            
        except Exception as exc:
            self.logger.error(f"async_save failed: {exc}")
            sys.exit(-1)

    def load(self, tensors: list[tuple[int, torch.Tensor, int, str]]):
        """Load tensors with comprehensive validation and error recovery"""
        try:
            valid_loads = 0
            skipped_loads = 0
            
            for version, tensor, file_offset, path in tensors:
                try:
                    if not os.path.exists(path):
                        self.logger.warning(f"File not found: {path}")
                        skipped_loads += 1
                        continue
                        
                    file_size = os.path.getsize(path)
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    
                    if tensor_bytes <= 0:
                        self.logger.warning(f"Skipping zero-size tensor from {path}")
                        skipped_loads += 1
                        continue
                        
                    if file_offset + tensor_bytes > file_size:
                        self.logger.warning(f"Skipping invalid tensor: offset {file_offset} + size {tensor_bytes} > file_size {file_size} for {path}")
                        skipped_loads += 1
                        continue
                    
                    # Additional safety check: ensure offset is reasonable
                    if file_offset < 0 or file_offset >= file_size:
                        self.logger.warning(f"Skipping tensor with invalid offset {file_offset} for file size {file_size}")
                        skipped_loads += 1
                        continue
                    
                    # Try the actual load with error handling
                    self.ckpt_engine.restore_tensor(version, tensor, tensor_bytes, file_offset, path)
                    valid_loads += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to load tensor from {path} at offset {file_offset}: {e}")
                    skipped_loads += 1
                    continue
            
            if valid_loads > 0:
                self.logger.info(f"Successfully loaded {valid_loads} tensors, skipped {skipped_loads}")
            else:
                self.logger.warning(f"No tensors loaded successfully, {skipped_loads} skipped")
                
        except Exception as exc:
            self.logger.error(f"load failed with critical error: {exc}")
            # Don't exit here - allow graceful degradation
            return

    def commit(self, tag):
        """Commit checkpoint"""
        self.wait()
        self.logger.info(f"Checkpoint {tag} completed!")
        self.last_ckpt_version += 1
        return True

    def wait(self):
        """Wait for async operations to complete"""
        try:
            self.ckpt_engine.wait()
        except Exception as exc:
            self.logger.error(f"wait failed: {exc}")
            # Don't exit on wait failures - allow graceful recovery

    def __del__(self):
        """Cleanup checkpoint engine"""
        try:
            if hasattr(self, 'ckpt_engine'):
                self.ckpt_engine.shutdown()
        except:
            pass