m = -inf; d = 0.0; o = 0  # o 是向量
for K_blk, V_blk in blocks(K, V):
    logits = (q @ K_blk.T) / sqrt(dk) 
    m_blk = logits.max()
    m_new = max(m, m_blk)
    alpha = exp(m - m_new)
    p = exp(logits - m_new)            # 按元素
    d = d * alpha + p.sum()
    o = o * alpha + p @ V_blk
    m = m_new
return o / d
