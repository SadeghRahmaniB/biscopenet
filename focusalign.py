"""
FocusAlign: a co-attention projector that lets video (or video-crop) tokens
attend into text (or a salient-region "reduced" stream) before being handed
to the LLM, instead of a plain linear/MLP vision projector.

Usage: build a config object/namespace with `hidden_size` and
`mm_hidden_size` attributes, then:
    projector = CoAttentionProjector(config)
    vision_tokens, _, attn_scores = projector(video_features, text_embeds)
"""

import torch
from torch import nn
import math
class BertBiAttention(nn.Module):
    def __init__(self,num_attention_heads=8,hidden_size=512,dropout_rate=0.1):
        super(BertBiAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads) )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads) #每个头的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        # self.key1 = nn.Linear(hidden_size, self.all_head_size)
        # self.value1 = nn.Linear(hidden_size, self.all_head_size)
        # self.dropout1 = nn.Dropout(dropout_rate)
        # self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
    ):

        mixed_query_layer1 = self.query1(input_tensor1)
        # mixed_key_layer1 = self.key1(input_tensor1)
        # mixed_value_layer1 = self.value1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        # key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        # value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)

        # query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        # attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        # attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        # attention_scores1 = attention_scores1 * attention_mask1

        # attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        # attention_probs1 = self.dropout1(attention_probs1)

        # context_layer1 = torch.matmul(attention_probs1, value_layer1)
        # context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        # context_layer1 = context_layer1.view(*new_context_layer_shape1)

        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)

        attention_scores2 = attention_scores2 + attention_mask2
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = {
                "attn2": attention_probs2,
                # "queries2": query_layer2,
                # "keys1": key_layer1,
                # "attn1": attention_probs1,
                "querues1": query_layer1,
                "keys2": key_layer2,
            }

        context_layer1 = None
        return context_layer1, context_layer2, attn_data


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertBiOutput(nn.Module):
    def __init__(self, hidden_size=512,dropout_rate=0.1):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        # self.q_dense1 = nn.Linear(hidden_size, hidden_size)
        # self.q_dropout1 = nn.Dropout(dropout_rate)

        # self.dense2 = nn.Linear(hidden_size, hidden_size)
        # self.LayerNorm2 = nn.LayerNorm(hidden_size)
        # self.dropout2 = nn.Dropout(dropout_rate)

        # self.q_dense2 = nn.Linear(hidden_size, hidden_size)
        # self.q_dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)
        # context_state2 = self.dense2(hidden_states2)
        # context_state2 = self.dropout2(context_state2)
        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        # hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)
        hidden_states2 = None
        return hidden_states1, hidden_states2

class BertImageIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertImageOutput(nn.Module):
    def __init__(self, hidden_size=512,dropout_rate=0.1):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, hidden_size=512,dropout_rate=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Co_attention_block(nn.Module):
    def __init__(self,num_attention_heads=6,hidden_size=512,dropout_rate=0.1):
        super(Co_attention_block, self).__init__()
        self.biattention = BertBiAttention(num_attention_heads,hidden_size,dropout_rate)

        self.biOutput = BertBiOutput(hidden_size,dropout_rate)

        self.v_intermediate = BertImageIntermediate(hidden_size)
        self.v_output = BertImageOutput(hidden_size,dropout_rate)

        # self.t_intermediate = BertIntermediate(hidden_size)
        # self.t_output = BertOutput(hidden_size,dropout_rate)

    def forward(
        self,
        vision_input_tensor,
        vision_attention_mask,
        text_input_tensor,
        text_attention_mask,
    ):

        text_bi_output, vision_bi_output, co_attention_probs = self.biattention(
            vision_input_tensor,
            vision_attention_mask,
            text_input_tensor,
            text_attention_mask,)
        vision_attention_output, text_attention_output = self.biOutput(
            vision_bi_output, vision_input_tensor, text_bi_output, text_input_tensor)
        vision_intermediate_output = self.v_intermediate(vision_attention_output)
        vision_layer_output = self.v_output(vision_intermediate_output, vision_attention_output)
        # text_intermediate_output = self.t_intermediate(text_attention_output)
        # text_layer_output = self.t_output(text_intermediate_output, text_attention_output)

        text_layer_output = None
        return vision_layer_output, text_layer_output, co_attention_probs
if __name__ == "__main__":
    co_attention_block = Co_attention_block(hidden_size=512,num_attention_heads=8,dropout_rate=0.1)
    vision_input_tensor = torch.ones([1, 8, 512])
    vision_mask = torch.ones([vision_input_tensor.shape[0],vision_input_tensor.shape[1]])
    vision_mask = vision_mask.reshape(vision_mask.shape[0],1,1,vision_mask.shape[-1])
    text_input_tensor = torch.ones([1, 10, 512])
    text_mask = torch.ones([text_input_tensor.shape[0], text_input_tensor.shape[1]])
    vision_layer_output, text_layer_output, co_attention_probs = co_attention_block(vision_input_tensor,vision_mask,text_input_tensor,text_mask)
    print('vision_layer_output.size======',vision_layer_output.shape)
    print('text_layer_output.size==========',text_layer_output.shape)

    import thop
    import copy
    macs,params=thop.profile(copy.deepcopy(co_attention_block),inputs=(vision_input_tensor,vision_mask,text_input_tensor,text_mask,),verbose=False)
    macs,params=thop.clever_format([macs,params],"%.3f")
    print("vilbert-layer: macs:%s  params:%s"%(macs,params))

class CoAttentionProjector(nn.Module):  # Add nn.Module inheritance
    def __init__(self, config):
        super(CoAttentionProjector, self).__init__()

        hidden = config.hidden_size
        mm_hidden = config.mm_hidden_size

        # Pre-adaptation for video (replaces bare Linear)
        self.video_adapter = nn.Sequential(
            nn.LayerNorm(mm_hidden),
            nn.Linear(mm_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # (Optional) light norm for text embeddings (identity if not needed)
        # self.text_norm = nn.LayerNorm(hidden)

        self.flexi_co_attention = Co_attention_block(
            num_attention_heads=8,
            hidden_size=hidden,
            dropout_rate=0.1,
        )

        # Post-fusion FFN (helps re-align to LLM token statistics)
        self.post_ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )

        # Adapter for reduced (salient-region crop) features — same arch as video_adapter
        self.reduced_adapter = nn.Sequential(
            nn.LayerNorm(mm_hidden),
            nn.Linear(mm_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Learnable gates so model can ignore fusion early
        self.video_gate = nn.Parameter(torch.zeros(1))
        # self.text_gate = nn.Parameter(torch.zeros(1))

        # Factorized temporal positional encoding: a per-frame embedding plus a
        # per-patch (intra-frame) embedding, instead of one flat table indexed by
        # flattened (frame, patch) position. This lets the model learn "frame order"
        # as a small, separate signal (a handful of rows) rather than re-learning it
        # jointly with spatial position for every patch in every frame. Applied to
        # both the full-frame query stream and the reduced/cropped context stream,
        # so cross-attention has an explicit early-vs-late cue instead of having to
        # infer temporal order from appearance alone.
        self.use_pos_emb = True
        if self.use_pos_emb:
            self.max_frames = 64
            self.frame_pos = nn.Embedding(self.max_frames, hidden)
            self.patch_pos = nn.Parameter(torch.zeros(1, 256, hidden))  # grows at runtime if needed
            nn.init.zeros_(self.frame_pos.weight)

    def add_temporal_pos(self, x, num_frames):
        if not self.use_pos_emb or not num_frames:
            return x
        B, V, D = x.shape
        patches_per_frame = V // num_frames
        if patches_per_frame * num_frames != V:
            # Sequence doesn't factor cleanly into (num_frames, patches_per_frame); skip.
            return x
        if patches_per_frame > self.patch_pos.shape[1]:
            extra = torch.zeros(1, patches_per_frame - self.patch_pos.shape[1], D, device=x.device, dtype=x.dtype)
            self.patch_pos.data = torch.cat([self.patch_pos.data, extra], dim=1)
        frame_idx = torch.arange(num_frames, device=x.device).clamp(max=self.frame_pos.num_embeddings - 1)
        f_pos = self.frame_pos(frame_idx)  # [num_frames, D]
        p_pos = self.patch_pos[0, :patches_per_frame]  # [patches_per_frame, D]
        pos = (f_pos.unsqueeze(1) + p_pos.unsqueeze(0)).reshape(1, V, D)
        return x + pos.to(x.dtype)

    def build_additive_mask(self, mask, length, device, dtype):
        # Input mask may be [B,1,1,L] or [B,L]; expect 1 for keep, 0 for pad
        if mask is None:
            return torch.zeros(1, 1, length, device=device, dtype=dtype)  # adds 0 bias
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        # Convert to additive (0 for keep, -inf for pad)
        additive = (1.0 - mask.to(dtype=dtype)) * -1e4
        return additive

    def forward(
        self,
        video_spatio_temporal_features,
        inputs_embeds,
        reduced_features=None,
        vision_attention_mask=None,
        text_attention_mask=None,
        video_num_frames=None,
        reduced_num_frames=None,
        **kwargs
    ):
        # Adapt full-video features to LLM space
        video_spatio_temporal_features = video_spatio_temporal_features.to(inputs_embeds.dtype)
        vid_features = self.video_adapter(video_spatio_temporal_features)  # [B, V, D]
        vid_features = self.add_temporal_pos(vid_features, video_num_frames)

        B, V, _ = vid_features.shape

        # Context: salient-region crop features (preferred) or text embeddings (fallback)
        if reduced_features is not None:
            reduced_features = reduced_features.to(vid_features.dtype)
            context = self.reduced_adapter(reduced_features)  # [B, R, D]
            context = self.add_temporal_pos(context, reduced_num_frames)
            T = context.shape[1]
            text_add_mask = self.build_additive_mask(None, T, context.device, context.dtype)
        else:
            context = inputs_embeds
            T = inputs_embeds.shape[1]
            text_add_mask = self.build_additive_mask(text_attention_mask, T, inputs_embeds.device, inputs_embeds.dtype)

        vision_add_mask = self.build_additive_mask(vision_attention_mask, V, vid_features.device, vid_features.dtype)

        vision_attn_out, _, attention_scores = self.flexi_co_attention(
            vision_input_tensor=vid_features,
            vision_attention_mask=vision_add_mask,
            text_input_tensor=context,
            text_attention_mask=text_add_mask,
        )

        # Gated residual: full-video tokens enriched by salient-region context
        vision_attn_out = vid_features + self.video_gate.tanh() * vision_attn_out

        # Post-fusion FFN
        vision_attn_out = vision_attn_out + self.post_ffn(vision_attn_out)

        return vision_attn_out, None, attention_scores
    

