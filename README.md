# ratelimited-embedder

LangChain 速率控制向量化工具，支持批处理、自动降速、向量缓存、硬件建议。

## 安装

```bash
pip install ratelimited-embedder
```

开发模式：

```bash
git clone <repo>
cd ratelimited-embedder
pip install -e ".[dev]"
```

## 快速上手

```python
from langchain_ollama import OllamaEmbeddings
from ratelimited_embedder import RateControlledEmbedder

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# 直接在 RateControlledEmbedder 上配置缓存
embedder = RateControlledEmbedder(
    embeddings=embeddings,
    batch_size=16,
    delay=0.5,
    slow_threshold=2.0,
    cache_dir="./cache",
)

from langchain_core.documents import Document
chunks = [Document(page_content=f"文档片段 {i}") for i in range(100)]

vectorstore = embedder.build_vectorstore(chunks, save_path="faiss_index")
```

或者单独使用缓存包装器：

```python
from langchain_ollama import OllamaEmbeddings
from ratelimited_embedder import wrap_embeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
# 方式 1: 指定缓存目录（文件名自动生成为 vector_cache.db）
wrapped = wrap_embeddings(embeddings, cache_dir="./cache")
# 方式 2: 指定完整文件路径
wrapped = wrap_embeddings(embeddings, cache_path="vector_cache.db")
```

## 功能

- **速率控制** — 分批向量化，可配置 batch_size / delay
- **自动降速** — 单批耗时超过阈值时自动减半 batch_size、增大 delay
- **向量缓存** — SQLite + MD5，避免重复计算
- **硬件建议** — 根据内存/CPU 自动推荐参数
- **进度条** — tqdm 实时显示进度
- **流式预览** — 每批完成后输出统计

## API

### RateControlledEmbedder

```python
RateControlledEmbedder(
    embeddings,
    batch_size=16,
    delay=0.5,
    slow_threshold=2.0,
    cache=None,
    cache_path=None,
    cache_dir=None,
)
```

- `embeddings`: LangChain Embeddings 实例
- `batch_size`: 每批处理块数
- `delay`: 批次间等待秒数
- `slow_threshold`: 单批耗时超过此值触发自动降速
- `cache`: 传入已创建的 VectorCache 实例（优先级最高）
- `cache_path`: SQLite 文件完整路径
- `cache_dir`: 缓存目录，文件名自动生成为 vector_cache.db

方法：

- `build_vectorstore(chunks, save_path, progress_callback)` → FAISS 向量库
- `get_stats()` → dict（统计信息：total_chunks, degrade_count, avg_batch_time 等）
- `set_rate(batch_size, delay)` — 动态调整速率
- `get_rate_suggestion()` → dict（静态方法，获取硬件建议）

### wrap_embeddings

```python
wrap_embeddings(embeddings, cache_path=None, cache_dir=None, cache_class=None)
```

- `embeddings`: 任意 LangChain Embeddings 实例
- `cache_path`: SQLite 文件完整路径（优先级高于 cache_dir）
- `cache_dir`: 缓存目录，文件名自动生成为 vector_cache.db
- `cache_class`: 自定义缓存类

包装 LangChain Embeddings，自动启用 SQLite 向量缓存，对外接口不变。

### VectorCache

```python
VectorCache(db_path="vector_cache.db", cache_dir=None)
```

- `db_path`: SQLite 文件完整路径（优先级高于 cache_dir）
- `cache_dir`: 缓存目录，文件名自动生成为 vector_cache.db

方法：

- `get(text)` → list[float] | None — 查询单条缓存
- `put(text, vector, metadata)` — 写入单条缓存
- `get_batch(texts)` → (results, miss_indices) — 批量查询
- `put_batch(texts, vectors, metadata)` — 批量写入
- `stats()` → dict — 缓存统计
- `clear()` — 清空缓存

### get_hardware_suggestion

```python
from ratelimited_embedder import get_hardware_suggestion
info = get_hardware_suggestion()
# {'batch_size': 32, 'delay': 0.3, 'mem_gb': 16.0, 'mem_percent': 45, 'cpu_count': 8}
```

根据本机内存和 CPU 核心数推荐 batch_size 和 delay 参数。

### 类型定义

```python
from ratelimited_embedder import EmbeddingsProtocol, ProgressCallback
```

- `EmbeddingsProtocol`: LangChain Embeddings 接口协议，用于类型标注
- `ProgressCallback`: 进度回调类型 `Callable[[str], None]`

## License

Copyright (c) 2026 oi-star

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

A copy of the license is also available in the [LICENSE](LICENSE) file.
