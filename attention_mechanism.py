import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= False):
        super().__init__()
        assert d_out % num_heads == 0, "should be div"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        

        #implementing KV cache(simple)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.current_tok__pos = 0




    def forward(self,x, use_cache=False):
            b, num_tokens, d_in = x.shape
            queries = self.W_query(x)
            keys =self.W_key(x)
            values =  self.W_value(x)

            #add a head dimension to the shape
            keys_new = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values_new = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries_new = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            if use_cache:
                if self.cache_k is None:
                    self.cache_k, self.cache_v = keys_new, values_new
                else:
                    self.cache_k = torch.cat([self.cache_k, keys_new], dim=-1)
                    self.cache_v = torch.cat([self.cache_v, values_new], dim=-1)
            else:
                keys, values = keys_new, values_new       
            

            #transpose no. of tokens and head to get all matrix values in a certain head

            keys = keys.transpose(1,2)
            values = values.transpose(1,2)
            queries = queries_new.transpose(1,2)

            attn_scores = queries @ keys.transpose(2, 3)

            num_tokens_Q = queries.shape[-2]
            nums_tokens_k = keys.shape[-2]
            if use_cache:
                mask_simple = self.mask.bool()[self.current_tok__pos:self.current_tok__pos+num_tokens_Q, :nums_tokens_k]
            
            else:
                mask_simple = self.mask.bool()[:num_tokens_Q, :nums_tokens_k]

            mask_simple = self.mask.bool()[:num_tokens, :num_tokens]

            attn_scores.masked_fill_(mask_simple, -torch.inf)
            attn_weights = torch.softmax(attn_scores/keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1,2) #reformatted context vectors
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # concat the context vector

            context_vec = self.out_proj(context_vec) # optional projection for taking out importance from each attention head

            return context_vec


    def reset_cache(self):
        self.cache_k , self.cache_v = None, None
        self.current_tok__pos  = 0


            

        





