from deepspeed.runtime.engine import DeepSpeedEngine
from DataStates.checkpointing import Checkpointing

class DataStatesEngine(DeepSpeedEngine):
    """DeepSpeed engine with DataStates checkpointing integration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize DataStates checkpointing
        if hasattr(self.config, 'checkpoint') and self.config.checkpoint.get('use_datastates', False):
            datastates_config = self.config.checkpoint.get('datastates_config', {})
            self.datastates_checkpointing = Checkpointing(datastates_config, rank=self.global_rank)
        else:
            self.datastates_checkpointing = None
    
    def _get_checkpoint_state_dict(self, tag, client_state=None):
        """Get checkpoint state dict with DataStates support"""
        state_dict = super()._get_checkpoint_state_dict(tag, client_state)
        
        # Add DataStates metadata if needed
        if self.datastates_checkpointing:
            state_dict['datastates_metadata'] = {
                'layer_importance': self.datastates_checkpointing.layer_importance_scores,
                'selected_layers': self.datastates_checkpointing.selected_layers_cache
            }
        
        return state_dict