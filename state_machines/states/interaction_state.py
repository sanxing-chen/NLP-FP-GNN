from typing import List, Optional

import torch

from allennlp.nn import util
from allennlp.modules import Attention
from allennlp.modules.attention import AdditiveAttention, DotProductAttention

class InteractionState:
    """
    This class keeps track of all of interaction level encoder-RNN-related variables.
    """

    def __init__(self) -> None:
        self.turn_num = 0
        self.past_hidden_states = []
        self.past_cell_states = []
        self.encoder_outputs = []
        self.past_dec_hidden_states = []
        self.past_dec_cell_states = []
        self.attention = DotProductAttention()

    def update(self,
                last_hidden_state: torch.Tensor,
                last_cell_state: torch.Tensor,
                encoder_outputs: List[torch.Tensor]) -> None:
        self.past_hidden_states += [last_hidden_state]
        self.past_cell_states += [last_cell_state]
        self.encoder_outputs += [encoder_outputs]
    
    def update_dec(self,
                last_dec_hidden_state: torch.Tensor,
                last_dec_cell_state: torch.Tensor) -> None:
        self.turn_num += 1
        self.past_dec_hidden_states += [last_dec_hidden_state]
        self.past_dec_cell_states += [last_dec_cell_state]

    def get_last_lstm_states(self, encoder_output_dim):
        if self.turn_num == 0:
            return None
        
        return (self.past_hidden_states[-1][:encoder_output_dim].view(2,1,encoder_output_dim//2),
            self.past_cell_states[-1][:encoder_output_dim].view(2,1,encoder_output_dim//2))
    
    def get_last_dec_lstm_states(self):
        if self.turn_num == 0:
            return None
        
        return (self.past_dec_hidden_states[-1], self.past_dec_cell_states[-1])

    def get_past_lstm_context(self, query: torch.Tensor, w):
        if self.turn_num == 0:
            return torch.zeros_like(query)
        query = w(query)
        past_hidden = torch.stack(self.past_hidden_states)
        interation_attn_weights = self.attention(query.unsqueeze(0), 
            past_hidden.unsqueeze(0)).squeeze(0)
        attended_question = util.weighted_sum(past_hidden, interation_attn_weights)
        return attended_question
