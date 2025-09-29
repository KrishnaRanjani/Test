import torch
from concurrent.futures import ThreadPoolExecutor
import time
from collections import OrderedDict, deque
import sys
import os
import glob
from typing import Union, Dict, Set
import pickle
import json
import ctypes
import numpy as np
from datastates.ckpt import CkptEngine
from .helper import parse_config, get_checkpoint_version, HOST_CACHE_SIZE, CKPT_PARSER_THREADS
from datastates.utils import get_logger

def extract_layer_id(param_name):
    """Extract layer ID from parameter name"""
    if 'layers.' in param_name:
        parts = param_name.split('.')
        try:
            layer_idx = parts.index('layers')
            if layer_idx + 1 < len(parts) and parts[layer_idx + 1].isdigit():
                return f"layer_{parts[layer_idx + 1].zfill(2)}"
        except:
            pass
    
    if 'embed' in param_name:
        return "embed_tokens"
    elif 'norm' in param_name or 'lm_head' in param_name:
        return "lm_head"
    else:
        return "other"

class Checkpointing:
    def __init__(self, runtime_config={}, rank=0):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        self.rank = int(rank)
        datastates_config = parse_config(runtime_config)
        
        host_cache_size = int(datastates_config[HOST_CACHE_SIZE]*(1<<30))
        cuda_device = int(torch.cuda.current_device())
        
        self.ckpt_engine = CkptEngine(host_cache_size, cuda_device, self.rank)   
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.logger = get_logger(__name__)
        self.last_ckpt_version = -1

        required_keys = ['warmup_steps', 'partial_update_mode', 'full_checkpoint_interval', 'partial_checkpoint_interval', 'update_budget_ratio']
        for key in required_keys:
            if key not in datastates_config:
                self.logger.warning(f"Missing config key {key}, using default")

        self.warmup_steps = datastates_config.get('warmup_steps', 100)
        self.partial_update_mode = datastates_config.get('partial_update_mode', False)
        self.full_checkpoint_interval = datastates_config.get('full_checkpoint_interval', 500)
        self.partial_checkpoint_interval = datastates_config.get('partial_checkpoint_interval', 100)
        self.update_budget_ratio = datastates_config.get('update_budget_ratio', 0.3)
        
        self.training_start_step = None
        self.warmup_complete = False
        self.last_full_checkpoint_step = None
        
        self.baseline_gradients = {}
        self.current_gradients = {}
        self.layer_importance_scores = {}
        self.selected_layers_cache = None
        self.available_layers_cache = None
        
        self.importance_strategy = datastates_config.get('importance_strategy', 'gradient_norm_diff')
        self.param_mapping = {}

    def classify_parameter_type(self, param_name):
        """Classify parameters by optimizer state criticality"""
        if param_name.startswith('model.'):
            param_name = param_name[6:]
        
        if any(keyword in param_name for keyword in ['bias', 'norm.weight', 'layernorm']):
            return 'reinitializable'
        
        return 'critical'

    def should_perform_checkpoint(self, current_step: int, resume_step: int = None) -> str:
        if self.training_start_step is None:
            self.training_start_step = resume_step if resume_step is not None else current_step
        
        actual_step = current_step
        warmup_end_step = self.training_start_step + self.warmup_steps
        warmup_mid_step = self.training_start_step + (self.warmup_steps // 2)
        
        if actual_step < warmup_mid_step:
            return "none"
        elif actual_step == warmup_mid_step:
            return "full_baseline"
        elif actual_step < warmup_end_step:
            return "none"
        elif actual_step == warmup_end_step:
            self.warmup_complete = True
            return "full_warmup_end"
        else:
            if self.last_full_checkpoint_step is None:
                self.last_full_checkpoint_step = warmup_end_step
            
            steps_since_last_full = actual_step - self.last_full_checkpoint_step
            
            if steps_since_last_full >= self.full_checkpoint_interval:
                return "full_scheduled"
            
            if (self.partial_update_mode and 
                steps_since_last_full % self.partial_checkpoint_interval == 0 and
                steps_since_last_full != 0):
                return "partial"
        
        return "none"

    def _compute_gradient_norms(self, model):
        gradient_norms = {}
        if model is not None:
            try:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient_norms[name] = param.grad.norm().item()
                    else:
                        gradient_norms[name] = 0.0
            except Exception as e:
                self.logger.error(f"Error computing gradient norms: {e}")
        return gradient_norms

    def _compute_layer_importance(self, model, checkpoint_type):
        if model is None:
            self.logger.warning("Model is None, cannot compute layer importance")
            return {}
        
        try:
            current_grads = self._compute_gradient_norms(model)
            
            if checkpoint_type == "full_baseline":
                layer_baseline = {}
                for name, grad_norm in current_grads.items():
                    layer_id = extract_layer_id(name)
                    if layer_id not in layer_baseline:
                        layer_baseline[layer_id] = 0.0
                    layer_baseline[layer_id] += grad_norm
                
                self.baseline_gradients = layer_baseline
                self.logger.info(f"Stored baseline gradients for {len(self.baseline_gradients)} layers")
                return {}
            
            elif checkpoint_type in ["full_warmup_end", "full_scheduled"]:
                layer_current = {}
                for name, grad_norm in current_grads.items():
                    layer_id = extract_layer_id(name)
                    if layer_id not in layer_current:
                        layer_current[layer_id] = 0.0
                    layer_current[layer_id] += grad_norm
                
                self.current_gradients = layer_current
                importance_scores = {}
                
                for layer_id in self.current_gradients:
                    if layer_id in self.baseline_gradients:
                        diff = abs(self.current_gradients[layer_id] - self.baseline_gradients[layer_id])
                        importance_scores[layer_id] = diff
                    else:
                        importance_scores[layer_id] = self.current_gradients[layer_id]
                
                self.layer_importance_scores = importance_scores
                self.logger.info(f"Computed layer importance for {len(importance_scores)} layers")
                
                return importance_scores
            
            return self.layer_importance_scores
        except Exception as e:
            self.logger.error(f"Error in _compute_layer_importance: {e}")
            return {}

    def _select_important_layers(self):
        """Select important layers with critical layers always included"""
        if not self.layer_importance_scores:
            self.logger.warning("No layer importance scores available for selection")
            return {"embed_tokens", "lm_head"}
        
        critical_layers = {"lm_head", "embed_tokens"}
        
        remaining_layers = {k: v for k, v in self.layer_importance_scores.items() 
                           if k not in critical_layers}
        
        if remaining_layers:
            total_layers = len(self.layer_importance_scores)
            target_count = max(1, int(total_layers * self.update_budget_ratio))
            remaining_count = max(0, target_count - len(critical_layers))
            
            if remaining_count > 0:
                sorted_layers = sorted(remaining_layers.items(), key=lambda x: x[1], reverse=True)
                selected_layers = [layer_id for layer_id, _ in sorted_layers[:remaining_count]]
                critical_layers.update(selected_layers)
        
        self.logger.info(f"Selected {len(critical_layers)} layers: {sorted(critical_layers)}")
        return critical_layers

    def create_optimizer_param_mapping(self, model, optimizer):
        """FIXED: Create stable parameter name mapping"""
        param_mapping = {}
        if not model or not optimizer:
            return param_mapping
        
        # Create parameter name to index mapping for the current optimizer
        param_idx = 0
        for group in optimizer.param_groups:
            for param in group['params']:
                for name, model_param in model.named_parameters():
                    if param is model_param:
                        param_mapping[param_idx] = name
                        param_idx += 1
                        break
        
        return param_mapping

    def save_optimizer_state_properly(self, optimizer, model):
        """FIXED: Save optimizer state using parameter names as stable keys"""
        if not optimizer or not model:
            return {}
        
        # Get the full optimizer state dict
        opt_state_dict = optimizer.state_dict()
        
        # Create parameter name to index mapping for the CURRENT optimizer
        param_name_to_idx = {}
        idx_to_param_name = {}
        
        param_idx = 0
        for group_idx, group in enumerate(optimizer.param_groups):
            for param in group['params']:
                # Find parameter name
                param_name = None
                for name, model_param in model.named_parameters():
                    if param is model_param:
                        param_name = name
                        break
                
                if param_name:
                    param_name_to_idx[param_name] = param_idx
                    idx_to_param_name[param_idx] = param_name
                    param_idx += 1
        
        # Convert state dict to use parameter names
        name_based_state = {
            'state': {},
            'param_groups': opt_state_dict['param_groups'],
            'param_name_mapping': idx_to_param_name
        }
        
        # Convert parameter states to use names
        for param_idx, state in opt_state_dict['state'].items():
            if param_idx in idx_to_param_name:
                param_name = idx_to_param_name[param_idx]
                name_based_state['state'][param_name] = state
        
        return name_based_state

    def load_optimizer_state_properly(self, saved_optimizer_state, current_optimizer, current_model):
        """FIXED: Load optimizer state by matching parameter names to current optimizer structure"""
        if not saved_optimizer_state or not current_optimizer or not current_model:
            return None
        
        # Create current parameter name to index mapping
        current_param_name_to_idx = {}
        param_idx = 0
        for group in current_optimizer.param_groups:
            for param in group['params']:
                # Find parameter name
                for name, model_param in current_model.named_parameters():
                    if param is model_param:
                        current_param_name_to_idx[name] = param_idx
                        param_idx += 1
                        break
        
        # Start with current optimizer's state dict structure
        current_state_dict = current_optimizer.state_dict()
        
        # Clear existing states and rebuild from saved data
        current_state_dict['state'] = {}
        
        # Restore states by matching parameter names
        saved_states = saved_optimizer_state.get('state', {})
        for param_name, saved_state in saved_states.items():
            if param_name in current_param_name_to_idx:
                current_param_idx = current_param_name_to_idx[param_name]
                current_state_dict['state'][current_param_idx] = saved_state
        
        # Restore parameter group hyperparameters
        saved_groups = saved_optimizer_state.get('param_groups', [])
        for i, current_group in enumerate(current_state_dict['param_groups']):
            if i < len(saved_groups):
                saved_group = saved_groups[i]
                # Restore all hyperparameters except 'params' list
                for key, value in saved_group.items():
                    if key != 'params':
                        current_group[key] = value
        
        return current_state_dict

    def flatten_optimizer_state(self, optimizer, param_mapping):
        """Convert optimizer state dict to flat tensor dict grouped by layers"""
        if not optimizer or not param_mapping:
            return {}
            
        opt_state = optimizer.state_dict()
        flat_optimizer_tensors = {}
        
        for param_id, state in opt_state.get('state', {}).items():
            if param_id in param_mapping:
                param_name = param_mapping[param_id]
                for state_key, tensor in state.items():
                    if torch.is_tensor(tensor):
                        opt_param_name = f"optimizer.{param_name}.{state_key}"
                        flat_optimizer_tensors[opt_param_name] = tensor
        
        return flat_optimizer_tensors

    def group_tensors_by_layer(self, state_dict, model, optimizer):
        """FIXED: Group tensors with proper optimizer state handling"""
        try:
            layer_groups = {}
            
            # Store optimizer reference for proper serialization
            self._current_optimizer = optimizer
            self._current_model = model
            
            # Save optimizer state properly
            if optimizer and model:
                properly_saved_optimizer = self.save_optimizer_state_properly(optimizer, model)
                if properly_saved_optimizer:
                    # Store as metadata (not tensor)
                    layer_groups['_optimizer_data'] = {
                        'model': [], 'optimizer': [], 
                        'critical_optimizer': [], 'reinit_optimizer': []
                    }
                    layer_groups['_optimizer_data']['optimizer'].append(('_optimizer_metadata', properly_saved_optimizer))
                    layer_groups['_optimizer_data']['critical_optimizer'].append(('_optimizer_metadata', properly_saved_optimizer))
            
            # Group model parameters
            all_tensors = {}
            if 'model' in state_dict:
                model_state = state_dict['model']
                for param_name, tensor in model_state.items():
                    if torch.is_tensor(tensor):
                        all_tensors[f"model.{param_name}"] = tensor
            
            # Group tensors by layer
            for param_name, tensor in all_tensors.items():
                layer_id = extract_layer_id(param_name)
                
                if layer_id not in layer_groups:
                    layer_groups[layer_id] = {
                        'model': [], 'optimizer': [], 
                        'critical_optimizer': [], 'reinit_optimizer': []
                    }
                
                if param_name.startswith('model.'):
                    layer_groups[layer_id]['model'].append((param_name, tensor))
            
            self.logger.debug(f"Grouped tensors into {len(layer_groups)} layers: {list(layer_groups.keys())}")
            return layer_groups
            
        except Exception as e:
            self.logger.error(f"Error in group_tensors_by_layer: {e}")
            return {}

    def reconstruct_optimizer_state(self, flat_optimizer_dict, param_mapping, complete_param_mapping=None):
        """FIXED: Use current optimizer structure instead of trying to reconstruct parameter IDs"""
        # This method is deprecated in favor of the new approach
        # Return empty state to trigger proper loading via load_optimizer_state_properly
        return {'state': {}, 'param_groups': []}

    def save_distributed_checkpoint(self, state_dict, base_path: str, current_step: int, checkpoint_type: str, model, optimizer):
        """FIXED: Save with proper optimizer state handling"""
        try:
            if checkpoint_type.startswith("full"):
                checkpoint_dir = f"{os.path.dirname(base_path)}/global_step_{current_step}"
            else:
                checkpoint_dir = f"{os.path.dirname(base_path)}/partial_step_{current_step}"
            
            os.makedirs(f"{checkpoint_dir}/model", exist_ok=True)
            os.makedirs(f"{checkpoint_dir}/optimizer", exist_ok=True)
            os.makedirs(f"{checkpoint_dir}/extra_states", exist_ok=True)
            
            layer_groups = self.group_tensors_by_layer(state_dict, model, optimizer)
            if not layer_groups:
                self.logger.error("Failed to group tensors by layer")
                return False
            
            if checkpoint_type.startswith("full"):
                selected_layers = set(layer_groups.keys())
                self.available_layers_cache = selected_layers.copy()
            else:
                selected_layers = self._select_important_layers()
                if not selected_layers:
                    self.logger.warning("No important layers selected for partial checkpoint")
                    return False
                # Always include optimizer data in any checkpoint
                if '_optimizer_data' in layer_groups:
                    selected_layers.add('_optimizer_data')
            
            async_list = []
            version = get_checkpoint_version(base_path, self.last_ckpt_version)
            
            metadata = {
                'parameter_mapping': {},
                'layer_groups': {},
                'checkpoint_type': checkpoint_type,
                'selected_layers': list(selected_layers),
                'format_version': '2.0',  # Mark as using new format
                'internal_state': {
                    'baseline_gradients': self.baseline_gradients,
                    'current_gradients': self.current_gradients,
                    'layer_importance_scores': self.layer_importance_scores,
                    'training_start_step': self.training_start_step,
                    'warmup_complete': self.warmup_complete,
                    'last_full_checkpoint_step': self.last_full_checkpoint_step
                }
            }
            
            for layer_id in selected_layers:
                if layer_id not in layer_groups:
                    continue
                    
                layer_data = layer_groups[layer_id]
                metadata['layer_groups'][layer_id] = []
                
                # Always save model tensors
                if layer_data['model'] and layer_id != '_optimizer_data':
                    model_path = f"{checkpoint_dir}/model/{layer_id}.distcp"
                    model_file_path = f"model/{layer_id}.distcp"
                    model_metadata = []
                    model_offset = 0
                    
                    for param_name, tensor in layer_data['model']:
                        if tensor is not None and tensor.numel() > 0:
                            staged_tensor = tensor.detach().contiguous()
                            tensor_size = tensor.numel() * tensor.element_size()
                            
                            async_list.append((version, staged_tensor, model_offset, model_path))
                            
                            model_metadata.append({
                                'param_name': param_name,
                                'offset': model_offset,
                                'size': tensor_size,
                                'shape': list(tensor.shape),
                                'dtype': str(tensor.dtype)
                            })
                            metadata['layer_groups'][layer_id].append(param_name)
                            model_offset += tensor_size
                    
                    if model_metadata:
                        metadata['parameter_mapping'][model_file_path] = model_metadata
                
                # Handle optimizer metadata specially
                if layer_id == '_optimizer_data' and layer_data['optimizer']:
                    for param_name, optimizer_metadata in layer_data['optimizer']:
                        if param_name == '_optimizer_metadata':
                            # Save optimizer metadata as JSON
                            optim_path = f"{checkpoint_dir}/optimizer/optimizer_state.json"
                            with open(optim_path, 'w') as f:
                                # Convert tensors to serializable format
                                serializable_metadata = self._make_serializable(optimizer_metadata)
                                json.dump(serializable_metadata, f, default=str, indent=2)
                            
                            metadata['optimizer_metadata_path'] = 'optimizer/optimizer_state.json'
            
            # Save extra states
            extra_states = {k: v for k, v in state_dict.items() 
                          if k != 'model' and not torch.is_tensor(v)}
            if extra_states:
                with open(f"{checkpoint_dir}/extra_states/extra_states.json", 'w') as f:
                    json.dump(extra_states, f, default=str)
            
            if async_list:
                self.ckpt_engine.async_save(async_list)
                self.ckpt_engine.wait()
                
                with open(f"{checkpoint_dir}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                self.logger.info(f"Saved {len(async_list)} tensors across {len(selected_layers)} layers (fixed optimizer handling)")
            else:
                self.logger.warning("No tensors to save")
                return False
            
            self._update_tracking_files(checkpoint_dir, checkpoint_type, selected_layers, current_step)
            return True
        except Exception as e:
            self.logger.error(f"Error in save_distributed_checkpoint: {e}", exc_info=True)
            return False

    def _make_serializable(self, obj):
        """Convert tensors and other non-serializable objects to serializable format"""
        if isinstance(obj, torch.Tensor):
            # Handle BFloat16 by converting to float32 first since numpy doesn't support BFloat16
            if obj.dtype == torch.bfloat16:
                tensor_data = obj.cpu().float().numpy().tolist()
                original_dtype = 'torch.bfloat16'
            elif obj.dtype == torch.float16:
                # Also handle float16 for completeness
                tensor_data = obj.cpu().float().numpy().tolist()
                original_dtype = 'torch.float16'
            else:
                tensor_data = obj.cpu().numpy().tolist()
                original_dtype = str(obj.dtype)
            
            return {
                '_tensor_data': tensor_data,
                '_tensor_shape': list(obj.shape),
                '_tensor_dtype': original_dtype
            }
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            # Handle objects with attributes
            try:
                return self._make_serializable(obj.__dict__)
            except:
                return str(obj)
        else:
            # For any other non-serializable objects, convert to string
            try:
                import json
                json.dumps(obj)  # Test if it's already serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)

    def _restore_from_serializable(self, obj):
        """Restore tensors from serializable format"""
        if isinstance(obj, dict) and '_tensor_data' in obj:
            data = torch.tensor(obj['_tensor_data'])
            shaped_tensor = data.reshape(obj['_tensor_shape'])
            
            # Restore original dtype
            original_dtype = obj['_tensor_dtype']
            if original_dtype == 'torch.bfloat16':
                shaped_tensor = shaped_tensor.bfloat16()
            elif original_dtype == 'torch.float16':
                shaped_tensor = shaped_tensor.half()
            elif hasattr(torch, original_dtype.split('.')[-1]):
                target_dtype = getattr(torch, original_dtype.split('.')[-1])
                shaped_tensor = shaped_tensor.to(target_dtype)
            
            return shaped_tensor
        elif isinstance(obj, dict):
            return {k: self._restore_from_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._restore_from_serializable(item) for item in obj)
        else:
            return obj

    def _update_tracking_files(self, checkpoint_dir, checkpoint_type, selected_layers, current_step):
        """Update tracking files"""
        try:
            base_dir = os.path.dirname(checkpoint_dir)
            
            with open(f"{base_dir}/latest", 'w') as f:
                f.write(os.path.basename(checkpoint_dir))
            
            with open(f"{base_dir}/latest_checkpointed_iteration.txt", 'w') as f:
                f.write(str(current_step))
            
            if checkpoint_type == "partial":
                with open(f"{base_dir}/partial_ckpt.txt", 'a') as f:
                    layers_str = ','.join(sorted(selected_layers))
                    f.write(f"{os.path.basename(checkpoint_dir)}:{layers_str}\n")
        except Exception as e:
            self.logger.error(f"Error updating tracking files: {e}")

    def _find_latest_full_checkpoint(self, base_dir):
        """Find most recent full checkpoint"""
        full_dirs = [d for d in os.listdir(base_dir) if d.startswith('global_step_')]
        if not full_dirs:
            raise FileNotFoundError("No full checkpoints found")
        
        latest_step = max(int(d.split('_')[-1]) for d in full_dirs)
        return f"{base_dir}/global_step_{latest_step}"

    def _find_latest_partial_only(self, base_dir, latest_full_dir):
        """Find only the most recent partial checkpoint newer than latest full"""
        full_step = int(os.path.basename(latest_full_dir).split('_')[-1])
        
        latest_partial = None
        latest_step = full_step
        
        for d in os.listdir(base_dir):
            if d.startswith('partial_step_'):
                partial_step = int(d.split('_')[-1])
                if partial_step > full_step and partial_step > latest_step:
                    latest_partial = f"{base_dir}/{d}"
                    latest_step = partial_step
        
        return latest_partial, latest_step if latest_partial else None

    def _load_checkpoint_dir(self, checkpoint_dir):
        """Load tensors using direct file read"""
        metadata_path = f"{checkpoint_dir}/metadata.json"
        if not os.path.exists(metadata_path):
            self.logger.warning(f"No metadata found in {checkpoint_dir}")
            return {}, {}
                
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata from {checkpoint_dir}: {e}")
            return {}, {}
        
        state_dict = {}
        failed_loads = 0
        total_loads = 0

        for file_path, tensor_list in metadata.get('parameter_mapping', {}).items():
            full_path = f"{checkpoint_dir}/{file_path}"
            if not os.path.exists(full_path):
                self.logger.warning(f"Missing file: {full_path}")
                failed_loads += len(tensor_list)
                continue

            try:
                file_size = os.path.getsize(full_path)
                for tensor_info in tensor_list:
                    total_loads += 1
                    param_name = tensor_info['param_name']
                    offset = tensor_info['offset']
                    size = tensor_info['size']
                    shape = tensor_info['shape']
                    dtype = getattr(torch, tensor_info['dtype'].split('.')[-1])

                    if offset < 0 or offset + size > file_size:
                        self.logger.warning(f"Invalid tensor bounds for {param_name}")
                        failed_loads += 1
                        continue

                    try:
                        with open(full_path, 'rb') as f:
                            f.seek(offset)
                            raw_data = f.read(size)
                        
                        if len(raw_data) != size:
                            self.logger.warning(f"Incomplete read for {param_name}: got {len(raw_data)}, expected {size}")
                            failed_loads += 1
                            continue
                        
                        tensor = torch.frombuffer(raw_data, dtype=dtype).reshape(shape).clone()
                        state_dict[param_name] = tensor
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load tensor {param_name}: {e}")
                        failed_loads += 1
                        continue
            except Exception as e:
                self.logger.error(f"Error processing file {full_path}: {e}")
                failed_loads += len(tensor_list)

        extra_states_path = f"{checkpoint_dir}/extra_states/extra_states.json"
        if os.path.exists(extra_states_path):
            try:
                with open(extra_states_path, 'r') as f:
                    extra_states = json.load(f)
                    state_dict.update(extra_states)
            except Exception as e:
                self.logger.warning(f"Failed to load extra states: {e}")

        success_rate = (total_loads - failed_loads) / max(total_loads, 1)
        if success_rate < 0.8:
            self.logger.warning(f"Low success rate ({success_rate:.1%}) loading {checkpoint_dir}")
            return {}, {}

        return state_dict, metadata

    def _restore_internal_state(self, checkpoint_dir):
        """Restore gradient tracking state"""
        if not checkpoint_dir:
            return
            
        metadata_path = f"{checkpoint_dir}/metadata.json"
        if not os.path.exists(metadata_path):
            return
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            internal_state = metadata.get('internal_state', {})
            self.baseline_gradients = internal_state.get('baseline_gradients', {})
            self.current_gradients = internal_state.get('current_gradients', {})
            self.layer_importance_scores = internal_state.get('layer_importance_scores', {})
            self.training_start_step = internal_state.get('training_start_step')
            self.warmup_complete = internal_state.get('warmup_complete', False)
            self.last_full_checkpoint_step = internal_state.get('last_full_checkpoint_step')
        except Exception as e:
            self.logger.error(f"Failed to restore internal state: {e}")

    def _reconstruct_final_state(self, merged_state, metadata, resume_step, full_checkpoint_path=None, model=None, optimizer=None):
        """FIXED: Reconstruct final state dict with proper optimizer handling"""
        final_state_dict = {}
        model_state = {}
        
        for param_name, tensor in merged_state.items():
            if param_name.startswith('model.'):
                model_param_name = param_name[6:]
                model_state[model_param_name] = tensor
            else:
                final_state_dict[param_name] = tensor
        
        final_state_dict['model'] = model_state
        
        # Handle optimizer state with new method
        if optimizer and model:
            # Check if we have the new format with proper optimizer metadata
            if metadata.get('format_version') == '2.0' and 'optimizer_metadata_path' in metadata:
                # Load using new format
                try:
                    checkpoint_dir = full_checkpoint_path if full_checkpoint_path else ""
                    optim_metadata_path = f"{checkpoint_dir}/{metadata['optimizer_metadata_path']}"
                    
                    if os.path.exists(optim_metadata_path):
                        with open(optim_metadata_path, 'r') as f:
                            saved_optimizer_state = json.load(f)
                        
                        # Restore from serializable format
                        saved_optimizer_state = self._restore_from_serializable(saved_optimizer_state)
                        
                        # Load using proper method
                        reconstructed_optimizer_state = self.load_optimizer_state_properly(
                            saved_optimizer_state, optimizer, model
                        )
                        
                        if reconstructed_optimizer_state:
                            final_state_dict['optimizer'] = reconstructed_optimizer_state
                            self.logger.info("Successfully loaded optimizer state using new format")
                        else:
                            self.logger.warning("Failed to load optimizer state - will use fresh optimizer")
                    else:
                        self.logger.warning(f"Optimizer metadata file not found: {optim_metadata_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load optimizer state: {e}")
            else:
                # Old format - skip optimizer state to avoid errors
                self.logger.warning("Old checkpoint format detected - skipping optimizer state")
        
        self._restore_internal_state(full_checkpoint_path if full_checkpoint_path else "")
        
        total_params = len(final_state_dict.get('model', {}))
        self.logger.info(f"Loaded {total_params} model parameters, resume step: {resume_step}")
        
        return final_state_dict

    def _load_optimized(self, checkpoint_base_dir, model=None, optimizer=None):
        """FIXED: Optimized loading with proper optimizer handling"""
        self.logger.info("Starting optimized distributed checkpoint loading...")
        
        latest_full = self._find_latest_full_checkpoint(checkpoint_base_dir)
        self.logger.info(f"Loading from full checkpoint: {latest_full}")
        
        # Check if this is new format before proceeding
        metadata_path = f"{latest_full}/metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('format_version') == '2.0':
            # Use new format loading
            return self._load_new_format(checkpoint_base_dir, model, optimizer)
        
        # For old format, load only full checkpoint to avoid corruption
        merged_state, metadata = self._load_checkpoint_dir(latest_full)
        if not merged_state:
            raise RuntimeError(f"Failed to load from full checkpoint: {latest_full}")
        
        step = int(os.path.basename(latest_full).split('_')[-1])
        return self._reconstruct_final_state(
            merged_state, 
            metadata, 
            step, 
            full_checkpoint_path=latest_full,
            model=model,
            optimizer=optimizer
        )

    def _load_new_format(self, checkpoint_base_dir, model=None, optimizer=None):
        """Load checkpoint using new format with proper optimizer handling"""
        latest_full = self._find_latest_full_checkpoint(checkpoint_base_dir)
        
        # Load checkpoint data
        state_dict, metadata = self._load_checkpoint_dir(latest_full)
        if not state_dict:
            raise RuntimeError(f"Failed to load from full checkpoint: {latest_full}")
        
        final_state_dict = {}
        
        # Load model state
        model_state = {}
        for param_name, tensor in state_dict.items():
            if param_name.startswith('model.'):
                model_param_name = param_name[6:]
                model_state[model_param_name] = tensor
            else:
                final_state_dict[param_name] = tensor
        
        final_state_dict['model'] = model_state
        
        # Load optimizer state using new method
        if optimizer and model and 'optimizer_metadata_path' in metadata:
            optim_metadata_path = f"{latest_full}/{metadata['optimizer_metadata_path']}"
            
            if os.path.exists(optim_metadata_path):
                with open(optim_metadata_path, 'r') as f:
                    saved_optimizer_state = json.load(f)
                
                # Restore from serializable format
                saved_optimizer_state = self._restore_from_serializable(saved_optimizer_state)
                
                # Load using proper method
                reconstructed_optimizer_state = self.load_optimizer_state_properly(
                    saved_optimizer_state, optimizer, model
                )
                
                if reconstructed_optimizer_state:
                    final_state_dict['optimizer'] = reconstructed_optimizer_state
                    self.logger.info("Successfully loaded optimizer state using new format")
                else:
                    self.logger.warning("Failed to reconstruct optimizer state")
        
        step = int(os.path.basename(latest_full).split('_')[-1])
        return final_state_dict

    def _load_fallback(self, checkpoint_base_dir, model=None, optimizer=None):
        """Fallback: load only from latest full checkpoint"""
        try:
            latest_full = self._find_latest_full_checkpoint(checkpoint_base_dir)
            self.logger.info(f"Fallback loading from: {latest_full}")
            
            state_dict, metadata = self._load_checkpoint_dir(latest_full)
            if not state_dict:
                raise RuntimeError(f"Failed to load from {latest_full}")
            
            step = int(os.path.basename(latest_full).split('_')[-1])
            return self._reconstruct_final_state(
                state_dict, 
                metadata, 
                step, 
                full_checkpoint_path=latest_full,
                model=model,
                optimizer=optimizer
            )
        except Exception as e:
            self.logger.error(f"Fallback load failed: {e}")
            raise

    def load(self, checkpoint_base_dir, model=None, optimizer=None):
        """FIXED: Load distributed checkpoint with proper optimizer handling"""
        try:
            return self._load_optimized(checkpoint_base_dir, model, optimizer)
        except Exception as e:
            self.logger.warning(f"Optimized load failed: {e}, falling back")
            return self._load_fallback(checkpoint_base_dir, model, optimizer)

    def save(self, state_dict, path: str, current_step: int = None, model=None, optimizer=None, resume_step: int = None, checkpoint_type: str = None):
        try:
            if checkpoint_type is None:
                checkpoint_type = self.should_perform_checkpoint(current_step, resume_step)
            
            if checkpoint_type == "none":
                return False

            if model is not None:
                self._compute_layer_importance(model, checkpoint_type)
            else:
                self.logger.warning("Model is None - cannot compute layer importance")

            self.logger.info(f"Performing {checkpoint_type} checkpoint at step {current_step}")
            
            result = self.save_distributed_checkpoint(state_dict, path, current_step, checkpoint_type, model, optimizer)
            
            if result:
                self.last_ckpt_version += 1
                self.wait()
                
                if checkpoint_type == "full_scheduled":
                    self.last_full_checkpoint_step = current_step
                elif checkpoint_type == "full_warmup_end":
                    self.last_full_checkpoint_step = current_step
                    
                if checkpoint_type in ["full_warmup_end", "full_scheduled"]:
                    if self.layer_importance_scores:
                        self.selected_layers_cache = self._select_important_layers()
                        
                self.logger.info(f"Checkpoint {checkpoint_type} completed successfully")
            else:
                self.logger.error(f"Checkpoint {checkpoint_type} failed")
            
            return result
        except Exception as e:
            self.logger.error(f"Error in save method: {e}", exc_info=True)
            return False

    def wait(self):
        try:
            self.ckpt_engine.wait()
        except Exception as e:
            self.logger.error(f"Error in wait: {e}")

    def __del__(self):
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass