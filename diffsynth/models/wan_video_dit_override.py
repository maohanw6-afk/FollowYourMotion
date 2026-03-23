import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0,time_scale=1.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta,time_scale=time_scale)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0,time_scale=1.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device)*time_scale, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
    def split_QKV(self,heads_pos):
        head_dim=self.head_dim
        # num_head=self.num_heads
        self.heads_pos=heads_pos#0: temporal  1:spatial

        q=self.q.weight.data# out_dim in_dim
        k=self.k.weight.data
        v=self.v.weight.data
        q_temp=[]
        q_spa=[]
        k_temp = []
        k_spa = []
        v_temp = []
        v_spa = []
        for i, t in enumerate(self.heads_pos):
            chunk_q=q[i*head_dim:(i+1)*head_dim]
            chunk_k=k[i * head_dim:(i + 1) * head_dim]
            chunk_v=v[i * head_dim:(i + 1) * head_dim]
            if t==0:
                q_temp.append(chunk_q)
                k_temp.append(chunk_k)
                v_temp.append(chunk_v)
            else:
                q_spa.append(chunk_q)
                k_spa.append(chunk_k)
                v_spa.append(chunk_v)
        q_temp =torch.cat(q_temp)
        q_spa =torch.cat(q_spa)
        k_temp =torch.cat(k_temp)
        k_spa = torch.cat(k_spa)
        v_temp = torch.cat(v_temp)
        v_spa = torch.cat(v_spa)

        dim_tem,in_dim=q_temp.shape[0],q_temp.shape[1]
        dim_spa=q_spa.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        # 同时指定设备和数据类型
        self.q_temporal = nn.Linear(in_dim, dim_tem).to(device=device, dtype=dtype)
        self.q_spatial = nn.Linear(in_dim, dim_spa).to(device=device, dtype=dtype)
        self.k_temporal = nn.Linear(in_dim, dim_tem).to(device=device, dtype=dtype)
        self.k_spatial = nn.Linear(in_dim, dim_spa).to(device=device, dtype=dtype)
        self.v_temporal = nn.Linear(in_dim, dim_tem).to(device=device, dtype=dtype)
        self.v_spatial = nn.Linear(in_dim, dim_spa).to(device=device, dtype=dtype)

        self.q_temporal.weight.data=q_temp
        self.q_spatial.weight.data=q_spa
        self.k_temporal.weight.data=k_temp
        self.k_spatial.weight.data=k_spa
        self.v_temporal.weight.data = v_temp
        self.v_spatial.weight.data = v_spa

        del self.q,self.k,self.v
        self.requires_grad_(False)

    def mix_heads(self,temporal,spatial):
        #shape B S D
        heads_pos=self.heads_pos
        head_dim = self.head_dim
        out=[]
        ind_t=0
        ind_s=0
        for i,t in enumerate(heads_pos):
            if t==0:#temporal
                out.append(temporal[:,:,ind_t*head_dim:(ind_t+1)*head_dim])
                ind_t+=1
            else:
                out.append(spatial[:,:,ind_s * head_dim:(ind_s + 1) * head_dim])
                ind_s += 1
        return torch.cat(out,dim=-1)


    def forward(self, x, freqs):
        if getattr(self,'q',None) is not None:
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            x = flash_attention(
                q=rope_apply(q, freqs, self.num_heads),
                k=rope_apply(k, freqs, self.num_heads),
                v=v,
                num_heads=self.num_heads
            )
            return self.o(x)
        else:
            q = self.norm_q(self.mix_heads(self.q_temporal(x),self.q_spatial(x)))
            k = self.norm_k(self.mix_heads(self.k_temporal(x),self.k_spatial(x)))
            v =self.mix_heads(self.v_temporal(x),self.v_spatial(x))
            x = flash_attention(
                q=rope_apply(q, freqs, self.num_heads),
                k=rope_apply(k, freqs, self.num_heads),
                v=v,
                num_heads=self.num_heads
            )
            return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q, k, v, num_heads=self.num_heads)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel_override(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim,time_scale=1)#按比例改 train/inference

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
    def split_attention(self,all_heads_type):
        self.all_heads_type=all_heads_type
        for i, blk in enumerate(self.blocks):
            blk.self_attn.split_QKV(all_heads_type[i])

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Tuple, Optional
# from einops import rearrange
# from .utils import hash_state_dict_keys
# try:
#     import flash_attn_interface
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_3_AVAILABLE = False

# try:
#     import flash_attn
#     FLASH_ATTN_2_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_2_AVAILABLE = False

# try:
#     from sageattention import sageattn
#     SAGE_ATTN_AVAILABLE = True
# except ModuleNotFoundError:
#     SAGE_ATTN_AVAILABLE = False
    
    
# def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
#     if compatibility_mode:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = F.scaled_dot_product_attention(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     elif FLASH_ATTN_3_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
#         x = flash_attn_interface.flash_attn_func(q, k, v)
#         x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
#     elif FLASH_ATTN_2_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
#         x = flash_attn.flash_attn_func(q, k, v)
#         x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
#     elif SAGE_ATTN_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = sageattn(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     else:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = F.scaled_dot_product_attention(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     return x


# def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
#     return (x * (1 + scale) + shift)


# def sinusoidal_embedding_1d(dim, position):
#     sinusoid = torch.outer(position.type(torch.float64), torch.pow(
#         10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
#     x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
#     return x.to(position.dtype)


# def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0,time_scale=1.0):
#     # 3d rope precompute
#     f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta,time_scale=time_scale)
#     h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
#     w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
#     return f_freqs_cis, h_freqs_cis, w_freqs_cis


# def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0,time_scale=1.0):
#     # 1d rope precompute
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
#                    [: (dim // 2)].double() / dim))
#     freqs = torch.outer(torch.arange(end, device=freqs.device)*time_scale, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis


# def rope_apply(x, freqs, num_heads):
#     x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
#     x_out = torch.view_as_complex(x.to(torch.float64).reshape(
#         x.shape[0], x.shape[1], x.shape[2], -1, 2))
#     x_out = torch.view_as_real(x_out * freqs).flatten(2)
#     return x_out.to(x.dtype)


# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))
#     def norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
#     def forward(self, x):
#         dtype = x.dtype
#         return self.norm(x.float()).to(dtype) * self.weight


# class SelfAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.o = nn.Linear(dim, dim)
#         self.norm_q = RMSNorm(dim, eps=eps)
#         self.norm_k = RMSNorm(dim, eps=eps)
#     def split_QKV(self,heads_pos):
#         head_dim=self.head_dim
#         # num_head=self.num_heads
#         self.heads_pos=heads_pos#0: temporal  1:spatial

#         q=self.q.weight.data# out_dim in_dim
#         k=self.k.weight.data
#         v=self.v.weight.data
#         o = self.o.weight.data
#         norm_q=self.norm_q.weight.data
#         norm_k=self.norm_k.weight.data

#         q_temp=[]
#         q_spa=[]
#         k_temp = []
#         k_spa = []
#         v_temp = []
#         v_spa = []
#         o_temp = []
#         o_spa = []
#         norm_q_t=[]
#         norm_q_s=[]
#         norm_k_t=[]
#         norm_k_s=[]
#         for i, t in enumerate(self.heads_pos):
#             chunk_q=q[i*head_dim:(i+1)*head_dim]
#             chunk_k=k[i * head_dim:(i + 1) * head_dim]
#             chunk_v=v[i * head_dim:(i + 1) * head_dim]
#             chunk_o = o[:, i * head_dim:(i + 1) * head_dim]
#             chunk_norm_q=norm_q[i * head_dim:(i + 1) * head_dim]
#             chunk_norm_k=norm_k[i * head_dim:(i + 1) * head_dim]
#             if t==0:
#                 q_temp.append(chunk_q)
#                 k_temp.append(chunk_k)
#                 v_temp.append(chunk_v)
#                 o_temp.append(chunk_o)
#                 norm_q_t.append(chunk_norm_q)
#                 norm_k_t.append(chunk_norm_k)
#             else:
#                 q_spa.append(chunk_q)
#                 k_spa.append(chunk_k)
#                 v_spa.append(chunk_v)
#                 o_spa.append(chunk_o)
#                 norm_q_s.append(chunk_norm_q)
#                 norm_k_s.append(chunk_norm_k)
#         q_temp =torch.cat(q_temp)
#         q_spa =torch.cat(q_spa)
#         k_temp =torch.cat(k_temp)
#         k_spa = torch.cat(k_spa)
#         v_temp = torch.cat(v_temp)
#         v_spa = torch.cat(v_spa)
#         o_temp = torch.cat(o_temp, dim=-1)
#         o_spa = torch.cat(o_spa, dim=-1)
#         norm_q_t.extend(norm_q_s)
#         norm_k_t.extend(norm_k_s)
#         norm_q=torch.cat(norm_q_t,dim=-1)
#         norm_k=torch.cat(norm_k_t,dim=-1)



#         dim_tem,in_dim=q_temp.shape[0],q_temp.shape[1]
#         dim_spa=q_spa.shape[0]
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         dtype = torch.bfloat16

#         # 同时指定设备和数据类型
#         self.q_temporal = nn.Linear(in_dim, dim_tem).to(device=device, dtype=dtype)
#         self.q_spatial = nn.Linear(in_dim, dim_spa).to(device=device, dtype=dtype)
#         self.k_temporal = nn.Linear(in_dim, dim_tem).to(device=device, dtype=dtype)
#         self.k_spatial = nn.Linear(in_dim, dim_spa).to(device=device, dtype=dtype)
#         self.v_temporal = nn.Linear(in_dim, dim_tem).to(device=device, dtype=dtype)
#         self.v_spatial = nn.Linear(in_dim, dim_spa).to(device=device, dtype=dtype)
#         self.o_temporal = nn.Linear(dim_tem, in_dim).to(device=device, dtype=dtype)
#         self.o_spatial=nn.Linear(dim_spa,in_dim).to(device=device, dtype=dtype)

#         self.q_temporal.weight.data=q_temp
#         self.q_spatial.weight.data=q_spa
#         self.k_temporal.weight.data=k_temp
#         self.k_spatial.weight.data=k_spa
#         self.v_temporal.weight.data = v_temp
#         self.v_spatial.weight.data = v_spa
#         self.o_temporal.weight.data=o_temp
#         self.o_spatial.weight.data=o_spa
#         self.norm_q.weight.data=norm_q
#         self.norm_k.weight.data=norm_k
#         self.temp_dim=dim_tem

#         del self.q,self.k,self.v,self.o
#         self.requires_grad_(False)

#     # def mix_heads(self,temporal,spatial):
#     #     #shape B S D
#     #     heads_pos=self.heads_pos
#     #     head_dim = self.head_dim
#     #     out=[]
#     #     ind_t=0
#     #     ind_s=0
#     #     for i,t in enumerate(heads_pos):
#     #         if t==0:#temporal
#     #             out.append(temporal[:,:,ind_t*head_dim:(ind_t+1)*head_dim])
#     #             ind_t+=1
#     #         else:
#     #             out.append(spatial[:,:,ind_s * head_dim:(ind_s + 1) * head_dim])
#     #             ind_s += 1
#     #     return torch.cat(out,dim=-1)


#     def forward(self, x, freqs):
#         if getattr(self,'q',None) is not None:
#             q = self.norm_q(self.q(x))
#             k = self.norm_k(self.k(x))
#             v = self.v(x)
#             x = flash_attention(
#                 q=rope_apply(q, freqs, self.num_heads),
#                 k=rope_apply(k, freqs, self.num_heads),
#                 v=v,
#                 num_heads=self.num_heads
#             )
#             return self.o(x)
#         else:
#             q = self.norm_q(torch.cat([self.q_temporal(x), self.q_spatial(x)], dim=-1))
#             k = self.norm_k(torch.cat([self.k_temporal(x), self.k_spatial(x)], dim=-1))
#             v = torch.cat([self.v_temporal(x), self.v_spatial(x)], dim=-1)
#             x = flash_attention(
#                 q=rope_apply(q, freqs, self.num_heads),
#                 k=rope_apply(k, freqs, self.num_heads),
#                 v=v,
#                 num_heads=self.num_heads
#             )

#             x=self.o_temporal(x[:,:,:self.temp_dim])+self.o_spatial(x[:,:,self.temp_dim:])
#             return  x


# class CrossAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads

#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.o = nn.Linear(dim, dim)
#         self.norm_q = RMSNorm(dim, eps=eps)
#         self.norm_k = RMSNorm(dim, eps=eps)
#         self.has_image_input = has_image_input
#         if has_image_input:
#             self.k_img = nn.Linear(dim, dim)
#             self.v_img = nn.Linear(dim, dim)
#             self.norm_k_img = RMSNorm(dim, eps=eps)

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         if self.has_image_input:
#             img = y[:, :257]
#             ctx = y[:, 257:]
#         else:
#             ctx = y
#         q = self.norm_q(self.q(x))
#         k = self.norm_k(self.k(ctx))
#         v = self.v(ctx)
#         x = flash_attention(q, k, v, num_heads=self.num_heads)
#         if self.has_image_input:
#             k_img = self.norm_k_img(self.k_img(img))
#             v_img = self.v_img(img)
#             y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
#             x = x + y
#         return self.o(x)


# class DiTBlock(nn.Module):
#     def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.ffn_dim = ffn_dim

#         self.self_attn = SelfAttention(dim, num_heads, eps)
#         self.cross_attn = CrossAttention(
#             dim, num_heads, eps, has_image_input=has_image_input)
#         self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
#         self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
#         self.norm3 = nn.LayerNorm(dim, eps=eps)
#         self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
#             approximate='tanh'), nn.Linear(ffn_dim, dim))
#         self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

#     def forward(self, x, context, t_mod, freqs):
#         # msa: multi-head self-attention  mlp: multi-layer perceptron
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
#             self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
#         input_x = modulate(self.norm1(x), shift_msa, scale_msa)
#         x = x + gate_msa * self.self_attn(input_x, freqs)
#         x = x + self.cross_attn(self.norm3(x), context)
#         input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
#         x = x + gate_mlp * self.ffn(input_x)
#         return x


# class MLP(torch.nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.proj = torch.nn.Sequential(
#             nn.LayerNorm(in_dim),
#             nn.Linear(in_dim, in_dim),
#             nn.GELU(),
#             nn.Linear(in_dim, out_dim),
#             nn.LayerNorm(out_dim)
#         )

#     def forward(self, x):
#         return self.proj(x)


# class Head(nn.Module):
#     def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
#         super().__init__()
#         self.dim = dim
#         self.patch_size = patch_size
#         self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
#         self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
#         self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

#     def forward(self, x, t_mod):
#         shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
#         x = (self.head(self.norm(x) * (1 + scale) + shift))
#         return x


# class WanModel_override(torch.nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         in_dim: int,
#         ffn_dim: int,
#         out_dim: int,
#         text_dim: int,
#         freq_dim: int,
#         eps: float,
#         patch_size: Tuple[int, int, int],
#         num_heads: int,
#         num_layers: int,
#         has_image_input: bool,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.freq_dim = freq_dim
#         self.has_image_input = has_image_input
#         self.patch_size = patch_size

#         self.patch_embedding = nn.Conv3d(
#             in_dim, dim, kernel_size=patch_size, stride=patch_size)
#         self.text_embedding = nn.Sequential(
#             nn.Linear(text_dim, dim),
#             nn.GELU(approximate='tanh'),
#             nn.Linear(dim, dim)
#         )
#         self.time_embedding = nn.Sequential(
#             nn.Linear(freq_dim, dim),
#             nn.SiLU(),
#             nn.Linear(dim, dim)
#         )
#         self.time_projection = nn.Sequential(
#             nn.SiLU(), nn.Linear(dim, dim * 6))
#         self.blocks = nn.ModuleList([
#             DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
#             for _ in range(num_layers)
#         ])
#         self.head = Head(dim, out_dim, patch_size, eps)
#         head_dim = dim // num_heads
#         self.freqs = precompute_freqs_cis_3d(head_dim,time_scale=1)#按比例改 train/inference

#         if has_image_input:
#             self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

#     def patchify(self, x: torch.Tensor):
#         x = self.patch_embedding(x)
#         grid_size = x.shape[2:]
#         x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
#         return x, grid_size  # x, grid_size: (f, h, w)

#     def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
#         return rearrange(
#             x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
#             f=grid_size[0], h=grid_size[1], w=grid_size[2], 
#             x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
#         )
#     def split_attention(self,all_heads_type):
#         self.all_heads_type=all_heads_type
#         for i, blk in enumerate(self.blocks):
#             blk.self_attn.split_QKV(all_heads_type[i])

#     def forward(self,
#                 x: torch.Tensor,
#                 timestep: torch.Tensor,
#                 context: torch.Tensor,
#                 clip_feature: Optional[torch.Tensor] = None,
#                 y: Optional[torch.Tensor] = None,
#                 use_gradient_checkpointing: bool = False,
#                 use_gradient_checkpointing_offload: bool = False,
#                 **kwargs,
#                 ):
#         t = self.time_embedding(
#             sinusoidal_embedding_1d(self.freq_dim, timestep))
#         t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
#         context = self.text_embedding(context)
        
#         if self.has_image_input:
#             x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
#             clip_embdding = self.img_emb(clip_feature)
#             context = torch.cat([clip_embdding, context], dim=1)
        
#         x, (f, h, w) = self.patchify(x)
        
#         freqs = torch.cat([
#             self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
#             self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
#             self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
#         ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
#         def create_custom_forward(module):
#             def custom_forward(*inputs):
#                 return module(*inputs)
#             return custom_forward

#         for block in self.blocks:
#             if self.training and use_gradient_checkpointing:
#                 if use_gradient_checkpointing_offload:
#                     with torch.autograd.graph.save_on_cpu():
#                         x = torch.utils.checkpoint.checkpoint(
#                             create_custom_forward(block),
#                             x, context, t_mod, freqs,
#                             use_reentrant=False,
#                         )
#                 else:
#                     x = torch.utils.checkpoint.checkpoint(
#                         create_custom_forward(block),
#                         x, context, t_mod, freqs,
#                         use_reentrant=False,
#                     )
#             else:
#                 x = block(x, context, t_mod, freqs)

#         x = self.head(x, t)
#         x = self.unpatchify(x, (f, h, w))
#         return x

#     @staticmethod
#     def state_dict_converter():
#         return WanModelStateDictConverter()
    
    
# class WanModelStateDictConverter:
#     def __init__(self):
#         pass

#     def from_diffusers(self, state_dict):
#         rename_dict = {
#             "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
#             "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
#             "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
#             "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
#             "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
#             "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
#             "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
#             "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
#             "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
#             "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
#             "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
#             "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
#             "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
#             "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
#             "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
#             "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
#             "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
#             "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
#             "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
#             "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
#             "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
#             "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
#             "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
#             "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
#             "blocks.0.norm2.bias": "blocks.0.norm3.bias",
#             "blocks.0.norm2.weight": "blocks.0.norm3.weight",
#             "blocks.0.scale_shift_table": "blocks.0.modulation",
#             "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
#             "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
#             "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
#             "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
#             "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
#             "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
#             "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
#             "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
#             "condition_embedder.time_proj.bias": "time_projection.1.bias",
#             "condition_embedder.time_proj.weight": "time_projection.1.weight",
#             "patch_embedding.bias": "patch_embedding.bias",
#             "patch_embedding.weight": "patch_embedding.weight",
#             "scale_shift_table": "head.modulation",
#             "proj_out.bias": "head.head.bias",
#             "proj_out.weight": "head.head.weight",
#         }
#         state_dict_ = {}
#         for name, param in state_dict.items():
#             if name in rename_dict:
#                 state_dict_[rename_dict[name]] = param
#             else:
#                 name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
#                 if name_ in rename_dict:
#                     name_ = rename_dict[name_]
#                     name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
#                     state_dict_[name_] = param
#         if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
#             config = {
#                 "model_type": "t2v",
#                 "patch_size": (1, 2, 2),
#                 "text_len": 512,
#                 "in_dim": 16,
#                 "dim": 5120,
#                 "ffn_dim": 13824,
#                 "freq_dim": 256,
#                 "text_dim": 4096,
#                 "out_dim": 16,
#                 "num_heads": 40,
#                 "num_layers": 40,
#                 "window_size": (-1, -1),
#                 "qk_norm": True,
#                 "cross_attn_norm": True,
#                 "eps": 1e-6,
#             }
#         else:
#             config = {}
#         return state_dict_, config
    
#     def from_civitai(self, state_dict):
#         if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
#             config = {
#                 "has_image_input": False,
#                 "patch_size": [1, 2, 2],
#                 "in_dim": 16,
#                 "dim": 1536,
#                 "ffn_dim": 8960,
#                 "freq_dim": 256,
#                 "text_dim": 4096,
#                 "out_dim": 16,
#                 "num_heads": 12,
#                 "num_layers": 30,
#                 "eps": 1e-6
#             }
#         elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
#             config = {
#                 "has_image_input": False,
#                 "patch_size": [1, 2, 2],
#                 "in_dim": 16,
#                 "dim": 5120,
#                 "ffn_dim": 13824,
#                 "freq_dim": 256,
#                 "text_dim": 4096,
#                 "out_dim": 16,
#                 "num_heads": 40,
#                 "num_layers": 40,
#                 "eps": 1e-6
#             }
#         elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
#             config = {
#                 "has_image_input": True,
#                 "patch_size": [1, 2, 2],
#                 "in_dim": 36,
#                 "dim": 5120,
#                 "ffn_dim": 13824,
#                 "freq_dim": 256,
#                 "text_dim": 4096,
#                 "out_dim": 16,
#                 "num_heads": 40,
#                 "num_layers": 40,
#                 "eps": 1e-6
#             }
#         else:
#             config = {}
#         return state_dict, config
