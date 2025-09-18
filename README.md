# 173_backend
No description

## 后端评测服务（Flask）

提供一个轻量后端，加载指定客户端模型并在DomainNet划分上评测Top-1准确率。

### 目录约定

- 预训练CLIP视觉骨干：`pretrained_models/clip-vit-base-patch16/`
- 客户端模型：`exp_models/Domainnet_ViT_fedsak_lda/client/client_model_*.pt`
- 数据集根目录：`/root/domainnet`

可通过环境变量覆盖默认路径：

- `MODELS_ROOT` (默认 `exp_models/Domainnet_ViT_fedsak_lda`)
- `DATA_ROOT` (默认 `/root/domainnet`)
- `CLIP_PRETRAINED_DIR` (默认 `pretrained_models/clip-vit-base-patch16`)
- `CLIP_PREPROCESSOR_JSON` (默认 `pretrained_models/clip-vit-base-patch16/preprocessor_config.json`)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python app.py
```

或指定端口：

```bash
PORT=8000 HOST=0.0.0.0 python app.py
```

### API

- GET `/health` 健康检查
- GET `/clients` 列出可用客户端ID
- POST `/evaluate` 评测

示例：

```bash
curl -X POST http://localhost:8000/evaluate \
	-H 'Content-Type: application/json' \
	-d '{"client_id": 1, "split": "test", "limit": 2000, "batch_size": 64}'
```

返回：

```json
{
	"client_id": 1,
	"split": "test",
	"samples": 2000,
	"correct": 1234,
	"accuracy": 0.617
}
```
