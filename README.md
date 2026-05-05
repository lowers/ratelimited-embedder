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
from ratelimited_embedder import RateControlledEmbedder, wrap_embeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
wrapped = wrap_embeddings(embeddings, cache_path="vector_cache.db")

embedder = RateControlledEmbedder(
    embeddings=wrapped,
    batch_size=16,
    delay=0.5,
    slow_threshold=2.0,
)

from langchain_core.documents import Document
chunks = [Document(page_content=f"文档片段 {i}") for i in range(100)]

vectorstore = embedder.build_vectorstore(chunks, save_path="faiss_index")
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
RateControlledEmbedder(embeddings, batch_size=16, delay=0.5, slow_threshold=2.0)
```

- `build_vectorstore(chunks, save_path, progress_callback)` → FAISS
- `get_stats()` → dict
- `set_rate(batch_size, delay)`
- `get_rate_suggestion()` → dict (静态方法)

### wrap_embeddings

```python
wrap_embeddings(embeddings, cache_path="vector_cache.db")
```

包装 LangChain Embeddings，自动启用 SQLite 向量缓存。

### VectorCache

```python
VectorCache(db_path="vector_cache.db")
```

- `get(text)` → list[float] | None
- `put(text, vector, metadata)`
- `stats()` → dict

## License

Copyright (c) 2025 oi-star

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