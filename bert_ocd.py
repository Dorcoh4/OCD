import torch.nn as nn
from copy import deepcopy
import transformers
import datasets
import torch
# bert = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
# sci_bert = transformers.AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=3)
# x = 3

class BertOriginal(nn.Module):
    def __init__(self, bert ):
        super(BertOriginal, self).__init__()
        self.bert = bert.cuda()

    def forward(self, input_ids, attention_mask):
        token_type_ids= None
        position_ids = None
        head_mask= None
        inputs_embeds = None
        encoder_hidden_states= None
        encoder_attention_mask= None
        past_key_values= None
        use_cache= None
        output_attentions = None
        output_hidden_states = None
        return_dict = None
        # output = self.bert.bert.embeddings(input)
        # output = self.bert.bert.encoder(output, attention_mask=attention_mask.cuda()).last_hidden_state

        output_attentions = output_attentions if output_attentions is not None else self.bert.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.bert.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.bert.bert.config.use_return_dict

        if self.bert.bert.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.bert.bert.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.bert.bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.bert.bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.bert.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.bert.get_head_mask(head_mask, self.bert.bert.config.num_hidden_layers)

        embedding_output = self.bert.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.bert.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        latent_in = deepcopy(sequence_output.detach())
        pooled_output = self.bert.bert.pooler(sequence_output) if self.bert.bert.pooler is not None else None
        latent = deepcopy(pooled_output.detach())

        # output = self.bert.bert.pooler(sequence_output)

        output = self.bert.dropout(pooled_output)
        output = self.bert.classifier(output,)
        return output,(latent,latent_in.squeeze())

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        self.bert.load_state_dict(state_dict, strict)

    def eval(self):
        self.bert.eval()