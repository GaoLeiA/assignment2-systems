import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,  # Pointers
    stride_qm, stride_qn,  # Strides
    # ... other args
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # TODO: Implement Triton Kernel for Forward Pass
    # 1. Load Q block to SRAM
    # 2. Loop over K, V blocks
    # 3. Compute QK^T
    # 4. Online Softmax
    # 5. Store Output
    pass

class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False):
        # 1. Launch Triton Kernel
        # 2. Save necessary tensors for backward
        pass

    @staticmethod
    def backward(ctx, do):
        # 1. Launch Triton Backward Kernel (optional/advanced)
        
        pass