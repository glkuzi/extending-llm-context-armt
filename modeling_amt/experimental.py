import torch
from modeling_rmt.language_modeling import *
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DistCausalLMOutputWithCrossAttentions(CausalLMOutputWithCrossAttentions):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    dist: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None


class Distillator(torch.nn.Module):
    def __init__(self, teacher_model, student_model, alpha_distil):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha_distil
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def forward(self, 
                input_ids, 
                labels=None, 
                labels_mask=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                input_segmented=False,
                sliding_window=False,
                ):
        with torch.no_grad():
            if self.training:
                if input_segmented:
                    raise NotImplementedError
                    n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]
                    teacher_output = self.teacher(
                        input_ids=torch.cat([input_ids[:, i] for i in range(n_segs)], dim=1) if input_ids is not None else None,
                        labels=torch.cat([labels[:, i] for i in range(n_segs)], dim=1)  if labels is not None else None, 
                        inputs_embeds=torch.cat([inputs_embeds[:, i] for i in range(n_segs)], dim=1) if inputs_embeds is not None else None, 
                        attention_mask=torch.cat([attention_mask[:, i] for i in range(n_segs)], dim=1) if attention_mask is not None else None, 
                        output_attentions=output_attentions, 
                        output_hidden_states=output_hidden_states
                    )
                else:
                    teacher_output = self.teacher(
                        input_ids=input_ids,
                        labels=labels, 
                        inputs_embeds=inputs_embeds, 
                        attention_mask=attention_mask, 
                        output_attentions=output_attentions, 
                        output_hidden_states=output_hidden_states
                    )
            else: 
                teacher_output = dict()
        student_output = self.student(
            input_ids=input_ids,
            labels=labels,
            labels_mask=labels_mask, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            input_segmented=input_segmented,
            sliding_window=sliding_window
        )
        
        if input_segmented:
            n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]
            labels = torch.cat([labels[:, i] for i in range(n_segs)], dim=1)
            if labels_mask is not None:
                labels_mask = torch.cat([labels_mask[:, i] for i in range(n_segs)], dim=1)
    
        out = self.process_outputs(teacher_output, student_output,
            labels=labels,
            labels_mask=labels_mask, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states)

        return out
                  
    def process_outputs(self, teacher_output, student_output, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        teacher_logits = teacher_output.logits if self.training else None
        student_logits = student_output.logits

        for (k, v) in student_output.items():
            out[k] = v
            
            teachers = teacher_output.get(k)
            # if teachers is not None:
            #     out[f'teacher_{k}'] = teachers

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_t_logits = teacher_logits[..., :-1, :].contiguous() if self.training else None

            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_t_logits = shift_t_logits.view(-1, shift_t_logits.size(-1)) if self.training else None
            
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
                flat_t_logits = flat_t_logits[shift_mask.view(-1)] if self.training else None

            dist_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

            log_sftmx_student = torch.log_softmax(flat_logits, dim=-1)  
            log_sftmx_teacher = torch.log_softmax(flat_t_logits, dim=-1) if self.training else None
            dist = dist_fct(log_sftmx_student, log_sftmx_teacher) if self.training else None
            # out['ce_loss'] = out['loss']
            ce_loss = out['loss']
            if self.training:
                # out['dist'] = dist
                # out['loss'] = (1 - self.alpha) * out['ce_loss'] + self.alpha * dist
                out['loss'] = (1 - self.alpha) * ce_loss + self.alpha * dist
                out['dist'] = dist
                out['ce_loss'] = ce_loss
                # recast out
                out = DistCausalLMOutputWithCrossAttentions(**out)
        else:
            out['loss'] = 0

        return out 

