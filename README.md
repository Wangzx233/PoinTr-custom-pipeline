# 点云补全 API 服务器

这是一个用于点云补全的API服务器，基于现有的点云补全代码构建。该API允许直接处理服务器上的点云文件和文件夹，无需上传或下载文件。

## 安装

1. 安装所需依赖:
```bash
pip install -r api_requirements.txt
```

## 运行服务器

使用以下命令启动API服务器:

```bash
python api_server.py --model_config <your_model_config_path> --model_checkpoint <your_model_checkpoint_path> --device cuda:0 --port 4011
```

参数说明:
- `--model_config`: 模型配置YAML文件路径（必需）
- `--model_checkpoint`: 预训练权重文件路径（必需）
- `--device`: 用于推理的设备 (默认: 'cuda:0')
- `--port`: API服务器端口 (默认: 4011)

## API 端点

### 健康检查

```
GET /health
```

返回服务器状态，用于确认服务器是否正常运行。

### 处理整个文件夹

```
POST /complete_folder
```

#### 请求参数 (JSON 格式):

```json
{
  "input_folder": "/path/to/input/folder",
  "output_folder": "/path/to/output/folder",
  "target_points": 8192,
  "sampling_method": "fps",
  "file_extension": ".ply"
}
```

- `input_folder`: 输入点云文件夹路径（服务器上的绝对路径）
- `output_folder`: 输出点云文件夹路径（服务器上的绝对路径）
- `target_points` (可选): 采样后的点数 (默认: 8192)
- `sampling_method` (可选): 采样方法, 可选 'fps', 'random', 或 'voxel' (默认: 'fps')
- `file_extension` (可选): 要处理的文件扩展名 (默认: '.ply')

#### 响应:

```json
{
  "total_files": 10,
  "successful": 9,
  "results": [
    {
      "file": "example1.ply",
      "status": "success",
      "output_path": "/path/to/output/folder/example1.ply"
    },
    {
      "file": "example2.ply",
      "status": "failed",
      "error": "错误描述"
    },
    ...
  ]
}
```

### 处理单个文件

```
POST /complete_file
```

#### 请求参数 (JSON 格式):

```json
{
  "input_file": "/path/to/input/file.ply",
  "output_file": "/path/to/output/file.ply",
  "target_points": 8192,
  "sampling_method": "fps"
}
```

- `input_file`: 输入点云文件路径（服务器上的绝对路径）
- `output_file`: 输出点云文件路径（服务器上的绝对路径）
- `target_points` (可选): 采样后的点数 (默认: 8192)
- `sampling_method` (可选): 采样方法, 可选 'fps', 'random', 或 'voxel' (默认: 'fps')

#### 响应:

```json
{
  "status": "success",
  "input_file": "/path/to/input/file.ply",
  "output_file": "/path/to/output/file.ply"
}
```

#### 错误响应:

```json
{
  "detail": "错误描述"
}
```

## 使用示例客户端

提供了一个简单的客户端脚本 `api_client.py`，可用于测试API:

### 处理文件夹:

```bash
python api_client.py --server_url http://localhost:4011 folder \
  --input_folder /path/to/input/folder \
  --output_folder /path/to/output/folder \
  --target_points 8192 \
  --sampling_method fps \
  --file_extension .ply
```

### 处理单个文件:

```bash
python api_client.py --server_url http://localhost:4011 file \
  --input_file /path/to/input/file.ply \
  --output_file /path/to/output/file.ply \
  --target_points 8192 \
  --sampling_method fps
```

## 部署到服务器

对于生产环境，服务已配置为使用uvicorn:

```bash
python api_server.py --model_config <your_model_config_path> --model_checkpoint <your_model_checkpoint_path> --device cuda:0 --port 4011
```

如果需要后台运行，可以使用nohup或screen:

```bash
nohup python api_server.py --model_config <your_model_config_path> --model_checkpoint <your_model_checkpoint_path> --device cuda:0 --port 4011 > api_server.log 2>&1 &
```

注意: 由于点云处理可能对资源要求较高，请确保服务器有足够的GPU内存和计算资源。 