# LLM Arithmetic Compressor 算法报告

## 1. 项目目标

本项目实现一个严格无损的文本压缩系统：

```text
input.txt -> compressed.bin + header.bin -> recovered.txt
```

核心思想是使用语言模型估计下一个 token 的概率分布，再用整数算术编码把真实 token 序列编码成紧凑 bitstream。解码端使用同一个模型和同一组参数重新生成概率分布，从 `compressed.bin` 中逐 token 恢复原文。

成功标准是：

```text
recovered.txt 与 input.txt 字节级完全一致
```

## 2. 当前算法总览

当前版本采用：

```text
Hugging Face causal LM
+ incremental cache
+ top-k escape probability model
+ integer arithmetic coding
+ 37-byte compact binary header
```

默认模型仍可使用 Qwen：

```text
Qwen/Qwen2.5-0.5B
Qwen/Qwen2.5-1.5B
Qwen/Qwen2.5-3B
```

项目也已经初步支持 Hugging Face Mamba causal LM，例如：

```text
state-spaces/mamba-130m-hf
```

Transformer 模型使用 `past_key_values` 增量缓存；Mamba 模型使用 `cache_params` 增量状态。

## 3. 压缩流程

压缩端流程如下：

```text
1. 读取 input.txt 字节
2. 按 UTF-8 解码成 text
3. 使用模型 tokenizer 得到 token_ids
4. 初始化模型 logit session
5. 对每个真实 token:
   a. 使用当前上下文/状态得到 next-token logits
   b. 从 logits 中取 top-k 候选 token
   c. 构造 top-k + escape 的整数频率表
   d. 若真实 token 在 top-k 中，编码其 top-k 内符号
   e. 若真实 token 不在 top-k 中，先编码 escape，再编码原 token id
   f. 将真实 token 推进模型 cache/state
6. 算术编码器输出 compressed.bin
7. 写入 compact header.bin
```

每个 token 的理想编码长度接近：

```text
-log2 P(token | context)
```

但当前不是全词表概率表，而是 top-k + escape 近似分布。

## 4. 解压流程

解压端流程如下：

```text
1. 读取 compressed.bin
2. 读取 header.bin
3. 加载同一个模型、tokenizer 和命令行参数
4. 初始化相同模型 logit session
5. 重复 token_count 次:
   a. 使用当前已解码上下文/状态得到 next-token logits
   b. 重建同一个 top-k + escape 频率表
   c. 从算术解码器中解出一个符号
   d. 如果符号不是 escape，映射回 top-k token
   e. 如果符号是 escape，再从均匀 token-id 表中解出原 token id
   f. 将解出的 token 推进模型 cache/state
6. tokenizer decode 得到 recovered text
7. UTF-8 编码回 bytes
8. 校验 original_size 和 CRC32
```

由于解压端必须重新生成完全相同的概率表，压缩与解压必须使用相同的：

```text
--model
--revision
--device
--dtype
--precision-bits
--top-k
--context-window
```

当前 compact header 不保存这些参数，它们是软件/命令行契约。

## 5. Top-k + Escape 概率模型

早期版本曾使用全词表概率表。Qwen 词表约为：

```text
151936 tokens
```

全词表量化每一步都要处理 15 万个 logits，并且每个 token 至少分配频率 1，会浪费概率质量。

当前版本改为：

```text
top-k model candidates + escape symbol
```

例如 `top_k = 4096` 时，每一步只构造：

```text
4096 个候选 token + 1 个 escape
```

编码规则：

```text
真实 token 在 top-k 内:
    算术编码 top-k 内符号

真实 token 不在 top-k 内:
    算术编码 escape
    再用均匀 token-id 表编码真实 token id
```

这种方案利用了语言模型的一个经验事实：


**真实下一个 token 通常排在模型预测的前若干名内**


优点：


1. 大幅减少每步频率表大小
2. 降低非零保底频率造成的概率质量浪费
3. 对长尾 token 仍然严格无损


缺点：


1. 如果真实 token 经常不在 top-k，escape 成本会升高
2. top-k 过小会损害压缩率
3. top-k 过大又会增加计算成本


## 6. 概率量化

语言模型输出的是浮点 logits。算术编码需要整数频率表。

当前量化流程：

```text
1. logits -> top-k token ids
2. 对 top-k logits 做稳定 softmax
3. escape 概率 = 1 - top-k 概率质量
4. 按 precision_bits 得到总频率:
   total = 2 ^ precision_bits
5. 每个符号至少分配 min_frequency = 1
6. 剩余频率按概率比例分配
7. 对小数余数做确定性补偿，保证总和固定
```

常用参数：

```text
precision_bits = 20 或 24
top_k = 4096
min_frequency = 1
```

提高 `precision_bits` 通常可以改善压缩率，但会增加整数范围和频率表处理成本。当前算术编码器使用 32-bit state，因此 `precision_bits` 不能无限增大。

## 7. 算术编码

算术编码维护一个整数区间：

```text
[low, high]
```

每编码一个符号，就按该符号在频率表中的累计区间缩小当前范围。

最终输出一串 bit：

```text
compressed.bin
```

注意：

```text
compressed.bin 中不保存概率
compressed.bin 中不保存 token rank
compressed.bin 中保存的是算术编码后的 bitstream
```

概率分布在解码时由模型重新生成。

## 8. Compact Header

当前 `header.bin` 固定 37 bytes。

字段为：

```text
magic/version
token_count
original_size
bit_length
CRC32
```

header 不再保存：

```text
model_name
tokenizer_name
revision
device/dtype
timing
compression ratio
完整 config
SHA256
```

这样做的原因是 header 对小文件影响很大。示例：

```text
payload_size = 482 B
旧 header_size ≈ 514 B
新 header_size = 37 B
```

完整包大小从：

```text
482 + 514 = 996 B
```

降为：

```text
482 + 37 = 519 B
```

## 9. 输出指标

压缩输出中主要看两组指标：

```text
payload_size  = compressed.bin 大小
payload_ratio = compressed.bin / original_size

package_size  = compressed.bin + header.bin
package_ratio = package_size / original_size
```

`payload_ratio` 表示去掉头之后的纯算术编码流压缩率。

`package_ratio` 表示真实可解码文件组合的压缩率。

例如一次 Qwen2.5-1.5B 实验：

```text
original_size = 7239 B
payload_size  = 482 B
header_size   = 37 B
package_size  = 519 B

payload_ratio = 6.66%
package_ratio = 7.17%
```

对比传统方法：

```text
gzip  ≈ 46.46%
bz2   ≈ 43.36%
lzma  ≈ 46.30%
zstd  ≈ 49.62%
```

在该样本上，LLM 算术编码显著优于这些传统通用压缩器。

## 10. 模型后端

### Transformer / Qwen

Transformer 后端使用 Hugging Face `AutoModelForCausalLM`。

增量推理使用：

```text
past_key_values
```

优点：

```text
1. 语言建模能力强
2. 压缩率好
3. Qwen 中文能力强
```

缺点：

```text
1. 长上下文总复杂度仍接近 O(N^2)
2. KV cache 随上下文增长
3. 压缩/解压速度慢于传统压缩器
```

### Mamba

Mamba 后端同样使用 Hugging Face `AutoModelForCausalLM`。

当模型配置中：

```text
model_type = mamba
```

项目会自动使用 Mamba session。

增量推理使用：

```text
cache_params
```

已验证模型：

```text
state-spaces/mamba-130m-hf
```

小样本 smoke test：

```text
compress ≈ 39.6 tok/s
decompress ≈ 42.8 tok/s
verify OK
```

当前环境缺少 Mamba fast kernels，Transformers 会回退到 sequential implementation。安装 `mamba-ssm`、`kernels` 或 `causal-conv1d` 后，Mamba 速度才更有代表性。

## 11. 复杂度分析

### Transformer

使用 KV cache 后，每一步只输入一个新 token，但该 token 仍需要 attend 历史 KV。

第 i 步成本近似：

```text
O(i)
```

总成本：

```text
O(1 + 2 + ... + N) = O(N^2)
```

### Mamba / State Space

Mamba 使用 recurrent/state-space 状态更新。

理想情况下，每步状态更新近似：

```text
O(1)
```

总成本：

```text
O(N)
```

这正是引入 Mamba 的原因：它可能在长文本压缩中显著降低复杂度。

## 12. 当前优势与限制

优势：

```text
1. 严格无损
2. 压缩率在中文样本上显著优于 gzip/bz2/lzma/zstd
3. header 已压缩到 37 bytes
4. 支持 Qwen Transformer 和 Mamba 后端
5. top-k escape 降低全词表处理成本
```

限制：

```text
1. 解压必须使用相同模型和参数
2. 当前不把模型权重计入压缩包大小
3. Transformer 后端速度仍慢
4. Mamba 需要 fast kernels 才能公平评估
5. CRC32 比 SHA256 更小但校验强度更弱
```

## 13. 后续研究方向

值得继续探索的方向：

```text
1. Mamba / RWKV / State Space LM 压缩率与速度对比
2. top_k 与 precision_bits 网格搜索
3. LoRA / 领域微调提升压缩率
4. rank coding + bzip2/zstd 作为高速模式
5. span-level / speculative residual compression
6. learned calibration module 优化 logits 概率校准
```

其中最有学术潜力的是：

```text
State-space language models for practical lossless text compression
```

核心假设：

```text
Transformer LLM 压缩率强但复杂度高；
state-space / recurrent LLM 可以用 O(N) streaming prior 获得接近的压缩率和更高速度。
```
