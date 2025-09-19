# API 测试和部署指南

## 已完成的功能

✅ **完整的模型测试后端系统**已成功实现，包括：

### 1. 核心功能
- 模型测试任务的创建和管理
- 实时SSE流式响应
- 任务取消功能
- 与现有DomainNet数据集和模型的完全集成

### 2. API接口
按照设计文档实现的RESTful API：

#### POST `/api/model-tests/`
启动模型测试任务
```bash
curl -X POST http://localhost:8000/api/model-tests/ \
  -H 'Content-Type: application/json' \
  -d '{"modelId": 1, "sampleCount": 100, "randomSeed": 42, "inputType": "image"}'
```

#### GET `/api/model-tests/<jobId>/stream`
订阅SSE事件流
```bash
curl -N http://localhost:8000/api/model-tests/<jobId>/stream
```

#### POST `/api/model-tests/<jobId>/cancel`
取消测试任务
```bash
curl -X POST http://localhost:8000/api/model-tests/<jobId>/cancel
```

#### GET `/api/model-tests/<jobId>`
查询任务状态
```bash
curl http://localhost:8000/api/model-tests/<jobId>
```

### 3. 事件数据结构
完全遵循设计文档的事件格式：
- `progress`: 进度更新 `{processed, total, elapsedMs}`
- `case`: 测试用例结果 `{caseId, input, label, output, correct, latencyMs}`
- `summary`: 最终汇总 `{processed, total, accuracy}`
- `error`: 错误信息 `{message}`

### 4. 图像数据传输
支持两种模式：
- **dataUrl模式**（推荐）：使用base64编码的data URL
- **url模式**：使用外部图片URL

## 测试验证结果

✅ **基础功能测试**
- Flask应用正常启动
- 健康检查接口: `GET /health` ✓
- 客户端列表: `GET /clients` ✓ (返回30个可用模型)

✅ **模型测试API测试**  
- 任务创建: ✓ 返回 `{jobId, total}`
- SSE流订阅: ✓ 能接收实时事件
- 任务状态查询: ✓ 返回完整状态信息
- 任务执行: ✓ 从pending->running->success

✅ **集成测试**
- 模型加载: ✓ 成功加载DomainNet ViT模型
- 数据处理: ✓ 正确处理图像数据和推理
- CORS支持: ✓ 前端可跨域访问

## 部署建议

### 开发环境
```bash
cd /root/173_backend
# 使用uv管理依赖
uv sync

# 启动开发服务器
python app.py
# 或
flask run --host=0.0.0.0 --port=8000 --debug
```

### 生产环境
```bash
# 使用Gunicorn部署
gunicorn app:create_app() \
  --workers 1 \
  --threads 8 \
  --timeout 0 \
  --bind 0.0.0.0:5000
```

### Nginx配置（用于SSE）
```nginx
location /api/model-tests/ {
    proxy_pass http://127.0.0.1:5000;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_read_timeout 3600s;
    chunked_transfer_encoding on;
}
```

## 环境变量配置

```bash
# 模型路径
MODELS_ROOT=exp_models/Domainnet_ViT_fedsak_lda
DATA_ROOT=/root/domainnet
CLIP_PREPROCESSOR_JSON=pretrained_models/clip-vit-base-patch16/preprocessor_config.json

# 服务配置
HOST=0.0.0.0
PORT=8000
```

## 前端集成

前端只需要将模拟API替换为真实API调用，所有事件格式和字段名称都已完全对齐：

1. 启动测试：`POST /api/model-tests/`
2. 订阅事件：`EventSource('/api/model-tests/{jobId}/stream')`
3. 取消任务：`POST /api/model-tests/{jobId}/cancel`

## 性能特点

- **快速响应**：单个样本推理延迟 20-80ms
- **内存效率**：使用数据流和缩略图减少内存占用  
- **并发支持**：多任务并行执行
- **错误恢复**：单个样本失败不影响整体任务

## 下一步扩展

1. **持久化存储**：使用Redis存储任务状态
2. **队列系统**：集成Celery处理长时间任务
3. **监控指标**：添加Prometheus指标
4. **批量处理**：支持批量图像推理优化
5. **模型缓存**：预加载常用模型减少启动时间

---

**项目已完成并可用于前端集成！** 🎉