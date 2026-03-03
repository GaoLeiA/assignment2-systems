import math
import torch


class FlashAttentionFunction(torch.autograd.Function):
    """
    FlashAttention2 的纯 PyTorch 实现。
    用 Python for 循环模拟 Tiling，用 Online Softmax 保证数值正确性。
    目的：验证算法逻辑，不追求速度。
    """

    @staticmethod
    def forward(ctx, q, k, v, block_mask=None, causal=False):
        """
        FlashAttention Forward Pass (分块 + Online Softmax)

        Args:
            q, k, v: (B, H, N, d) — Batch, Heads, SeqLen, HeadDim
            block_mask: Optional, 暂未使用
            causal: bool, 是否应用因果 (下三角) 掩码

        Returns:
            O: (B, H, N, d) — 注意力输出
        """
        B, H, N, d = q.shape
        BLOCK_SIZE = 64  # 可调，实际 GPU 上通常是 128 或 256

        # ==========================================
        # 初始化累加器 (都是按 Query 行维护)
        # ==========================================
        O = torch.zeros_like(q)                                        # (B, H, N, d) 未归一化输出
        L = torch.zeros(B, H, N, device=q.device, dtype=q.dtype)      # (B, H, N)   归一化因子 Σexp
        M = torch.full((B, H, N), float('-inf'), device=q.device, dtype=q.dtype)  # (B, H, N) 行最大值

        scale = 1.0 / math.sqrt(d)

        # ==========================================
        # 外层循环：遍历 Q 的块 (第 i 个块)
        # ==========================================
        for i in range(0, N, BLOCK_SIZE):
            i_end = min(i + BLOCK_SIZE, N)
            qi = q[:, :, i:i_end, :]  # (B, H, Br, d)

            # ==========================================
            # 内层循环：遍历 K/V 的块 (第 j 个块)
            # ==========================================
            for j in range(0, N, BLOCK_SIZE):
                j_end = min(j + BLOCK_SIZE, N)
                kj = k[:, :, j:j_end, :]  # (B, H, Bc, d)
                vj = v[:, :, j:j_end, :]  # (B, H, Bc, d)

                # --- Step 1: S_ij = Q_i @ K_j^T / sqrt(d) ---
                s_ij = torch.einsum('...qd,...kd->...qk', qi, kj) * scale  # (B, H, Br, Bc)

                # --- Step 1.5: Causal Mask (可选) ---
                if causal:
                    # row_idx: 全局 Query 位置, col_idx: 全局 Key 位置
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    causal_mask = row_idx >= col_idx  # 下三角为 True
                    s_ij = s_ij.masked_fill(~causal_mask, float('-inf'))

                # --- Step 2: Online Softmax 更新 ---
                m_prev = M[:, :, i:i_end]                       # (B, H, Br) 旧的行最大值
                block_max = s_ij.max(dim=-1).values             # (B, H, Br) 当前块的行最大值
                m_new = torch.maximum(m_prev, block_max)        # (B, H, Br) 新的全局行最大值

                # --- Step 3: P_ij = exp(S_ij - m_new) ---
                p_ij = torch.exp(s_ij - m_new.unsqueeze(-1))    # (B, H, Br, Bc) 安全的 exp

                # --- Step 4: Rescale 旧累加器 ---
                alpha = torch.exp(m_prev - m_new)               # (B, H, Br) 修正因子
                l_prev = L[:, :, i:i_end]                       # (B, H, Br) 旧的归一化因子
                l_new = l_prev * alpha + p_ij.sum(dim=-1)       # (B, H, Br) 新的归一化因子

                # --- Step 5: 更新输出 O (未归一化) ---
                o_prev = O[:, :, i:i_end, :]                    # (B, H, Br, d)
                # 旧输出乘以修正因子 + 新块贡献
                O[:, :, i:i_end, :] = o_prev * alpha.unsqueeze(-1) + p_ij @ vj

                # --- Step 6: 写回 M 和 L ---
                M[:, :, i:i_end] = m_new
                L[:, :, i:i_end] = l_new

        # ==========================================
        # 最终归一化: O = O_unnorm / L
        # ==========================================
        O = O / L.unsqueeze(-1)

        # 保存给 backward 用 (注意: 不存 P，这就是 FlashAttention 省显存的关键!)
        ctx.save_for_backward(q, k, v, O, L, M)
        ctx.causal = causal  # 非 Tensor 的元数据
        ctx.block_size = BLOCK_SIZE
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        FlashAttention Backward Pass (重计算 P，不存储 N×N 矩阵)

        核心思想:
            标准 Attention 的反向传播公式:
            - dV = P^T @ dO
            - dP = dO @ V^T
            - dS = P ⊙ (dP - D),  其中 D_i = Σ_j P_ij * dP_ij = Σ_j O_ij * dO_ij
            - dQ = dS @ K * scale
            - dK = dS^T @ Q * scale

            FlashAttention 的做法:
            对每一对 (i, j) 块，重新计算 P_ij = exp(S_ij - m_i) / l_i，
            然后累加 dQ, dK, dV。

        Args:
            dO: (B, H, N, d) — Loss 对 O 的梯度

        Returns:
            dQ, dK, dV: 各自形状与 q, k, v 相同
            None, None: 对应 block_mask 和 causal 的梯度 (不需要)
        """
        q, k, v, O, L, M = ctx.saved_tensors
        causal = ctx.causal
        BLOCK_SIZE = ctx.block_size

        B, H, N, d = q.shape
        scale = 1.0 / math.sqrt(d)

        # 初始化梯度
        dQ = torch.zeros_like(q)  # (B, H, N, d)
        dK = torch.zeros_like(k)  # (B, H, N, d)
        dV = torch.zeros_like(v)  # (B, H, N, d)

        # ==========================================
        # 预计算 D_i = rowsum(dO ⊙ O)
        # 这是 Softmax 反向传播的关键量
        # D_i = Σ_j dO_ij * O_ij (对每个 Query 行)
        # ==========================================
        D = (dO * O).sum(dim=-1)  # (B, H, N)

        # ==========================================
        # 分块循环：与 Forward 结构完全相同
        # ==========================================
        for i in range(0, N, BLOCK_SIZE):
            i_end = min(i + BLOCK_SIZE, N)
            qi = q[:, :, i:i_end, :]       # (B, H, Br, d)
            doi = dO[:, :, i:i_end, :]     # (B, H, Br, d)
            li = L[:, :, i:i_end]          # (B, H, Br)
            mi = M[:, :, i:i_end]          # (B, H, Br)
            di = D[:, :, i:i_end]          # (B, H, Br)

            for j in range(0, N, BLOCK_SIZE):
                j_end = min(j + BLOCK_SIZE, N)
                kj = k[:, :, j:j_end, :]   # (B, H, Bc, d)
                vj = v[:, :, j:j_end, :]   # (B, H, Bc, d)

                # --- 重计算 S_ij 和 P_ij (这就是 Recomputation!) ---
                s_ij = torch.einsum('...qd,...kd->...qk', qi, kj) * scale  # (B, H, Br, Bc)

                # Causal Mask (必须和 Forward 一致!)
                if causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    causal_mask = row_idx >= col_idx
                    s_ij = s_ij.masked_fill(~causal_mask, float('-inf'))

                # 重计算 P_ij = exp(S_ij - m_i) / l_i
                # 注意: Forward 中 O 最后除以了 L，所以这里 P 也要除以 L
                p_ij = torch.exp(s_ij - mi.unsqueeze(-1)) / li.unsqueeze(-1)  # (B, H, Br, Bc)

                # --- dV_j += P_ij^T @ dO_i ---
                # dV_jk = Σ_q P_qk * dO_qd → einsum: (Br,Bc)^T @ (Br,d) = (Bc,d)
                dV[:, :, j:j_end, :] += torch.einsum('...qk,...qd->...kd', p_ij, doi)

                # --- dP_ij = dO_i @ V_j^T ---
                dp_ij = torch.einsum('...qd,...kd->...qk', doi, vj)  # (B, H, Br, Bc)

                # --- dS_ij = P_ij ⊙ (dP_ij - D_i) ---
                # 这是 Softmax 的反向传播公式:
                # ∂L/∂S_ij = P_ij * (∂L/∂P_ij - D_i)
                ds_ij = p_ij * (dp_ij - di.unsqueeze(-1))  # (B, H, Br, Bc)

                # --- dQ_i += dS_ij @ K_j * scale ---
                dQ[:, :, i:i_end, :] += torch.einsum('...qk,...kd->...qd', ds_ij, kj) * scale

                # --- dK_j += dS_ij^T @ Q_i * scale ---
                dK[:, :, j:j_end, :] += torch.einsum('...qk,...qd->...kd', ds_ij, qi) * scale

        # 返回 5 个值，对应 forward 的 5 个输入 (q, k, v, block_mask, causal)
        return dQ, dK, dV, None, None


# Helper function exposed to adapters
def flash_attention_forward(q, k, v, causal=False):
    return FlashAttentionFunction.apply(q, k, v, None, causal)