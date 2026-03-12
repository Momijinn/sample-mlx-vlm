# sample-mlx-vlm

[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) を使ってローカル推論する最小サーバーです。  
デフォルトモデルは `mlx-community/Qwen3.5-9B-4bit` です。

提供API:

- `POST /v1/chat/completions` (OpenAI 互換)
- `POST /api/v1/chat` (LM Studio 互換)
- `POST /generate` (シンプル版)
- `GET /health`
- `GET /v1/models`
- `GET /api/v1/models` (LM Studio 互換)

## セットアップ

```bash
uv sync
```

## 起動

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

起動時には簡素なログで、

- 起動シーケンス開始
- モデル読み込み中
- 起動完了

が見えるようになっています。

モデルは既定で `mlx-community/Qwen3.5-9B-4bit` を使用します。  
変更したい場合のみ `MODEL_ID` を指定してください。

音声入力は `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` で確認済みです。

ダウンロードしたモデルや Hugging Face 系キャッシュは、既定でこのリポジトリ配下の `models/` に保存されます。これはアプリ起動時に設定され、以後のダウンロードに適用されます。

```bash
export MODEL_ID=mlx-community/Qwen3.5-9B-4bit
export MODEL_DIR=./models
export TRUST_REMOTE_CODE=false
export STRIP_THINKING_DEFAULT=false
export PRELOAD_MODEL_ON_STARTUP=true
```

- `MODEL_DIR` を指定すると、モデル保存先を変更できます。
- `TRUST_REMOTE_CODE=true` にすると、カスタム Hugging Face processor が必要なモデルも読み込めます。
- `STRIP_THINKING_DEFAULT=true` にすると、既定で思考過程除去を有効化します。
- `PRELOAD_MODEL_ON_STARTUP=false` にすると、モデル読み込みを初回リクエスト時まで遅延できます。
- 音声を使う場合は `export MODEL_ID=mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` を指定してください。
- Qwen3-Omni は大きいモデルなので、音声用途で使う場合は `PRELOAD_MODEL_ON_STARTUP=false` を推奨します。

## API 呼び出し例

### OpenAI互換 (`/v1/chat/completions`)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [
      {"role": "user", "content": "こんにちは"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "strip_thinking": true
  }'
```

画像入力例（`data_url` または `http(s)` URL）:

`data_url` を `curl` で送る例:

```bash
IMAGE_BASE64=$(base64 < sample.png | tr -d '\n')

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"mlx-community/Qwen3.5-9B-4bit\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          { \"type\": \"text\", \"text\": \"この画像を説明して\" },
          { \"type\": \"image_url\", \"image_url\": { \"url\": \"data:image/png;base64,${IMAGE_BASE64}\" } }
        ]
      }
    ],
    \"max_tokens\": 256,
    \"strip_thinking\": true
  }"
```

画像URLを `curl` で送る例:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "この画像を説明して" },
          { "type": "image_url", "image_url": { "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png" } }
        ]
      }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

複数画像URLを `curl` で送る例:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "この2枚の共通点と違いを教えて" },
          { "type": "image_url", "image_url": { "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png" } },
          { "type": "image_url", "image_url": { "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png" } }
        ]
      }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

※ 上の例では同じ画像を 2 回使っています。実運用では 2 枚目を別画像 URL に置き換えてください。

音声入力を `curl` で送る例:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "この音声の内容を要約して" },
          { "type": "input_audio", "input_audio": "./test/7463112128292362.mp3" }
        ]
      }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

動画URLを `curl` で送る例:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "この動画を要約して" },
          { "type": "video_url", "video_url": { "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" } }
        ]
      }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

ローカル動画ファイルを `file://` で送る例:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "この動画を要約して" },
          { "type": "video_url", "video_url": { "url": "file:///Users/yourname/Movies/sample.mp4" } }
        ]
      }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

### LM Studio互換 (`/api/v1/chat`)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "system_prompt": "あなたは有能なアシスタントです。",
    "input": "Type \"I love Qwen3.5\" backwards",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "strip_thinking": true
  }'
```

画像URLを `curl` で送る例:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "input": [
      { "type": "message", "content": "この画像を説明して" },
      { "type": "image", "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png" }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

ローカル画像を `data_url` で送る例:

```bash
IMAGE_BASE64=$(base64 < sample.png | tr -d '\n')

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"mlx-community/Qwen3.5-9B-4bit\",
    \"input\": [
      { \"type\": \"message\", \"content\": \"この画像を説明して\" },
      { \"type\": \"image\", \"data_url\": \"data:image/png;base64,${IMAGE_BASE64}\" }
    ],
    \"max_tokens\": 256,
    \"strip_thinking\": true
  }"
```

複数画像を `curl` で送る例:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "input": [
      { "type": "message", "content": "この2枚の共通点と違いを教えて" },
      { "type": "image", "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png" },
      { "type": "image", "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png" }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

※ こちらも例では同じ画像を 2 回使っています。2 枚目を別画像に変えれば、そのまま複数画像入力のサンプルとして使えます。

音声入力を `curl` で送る例:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
    "input": [
      { "type": "message", "content": "この音声の内容を要約して" },
      { "type": "audio", "url": "./test/7463112128292362.mp3" }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

動画URLを `curl` で送る例:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "input": [
      { "type": "message", "content": "この動画を要約して" },
      { "type": "video", "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

ローカル動画ファイルを `file://` で送る例:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
    "input": [
      { "type": "message", "content": "この動画を要約して" },
      { "type": "video", "url": "file:///Users/yourname/Movies/sample.mp4" }
    ],
    "max_tokens": 256,
    "strip_thinking": true
  }'
```

### シンプル (`/generate`)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "次の英語を逆順にしてください: I love Qwen3.5",
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "strip_thinking": true
  }'
```

## 注意

- 動画入力は `POST /v1/chat/completions` と `POST /api/v1/chat` で利用できます。
- 音声入力は `POST /v1/chat/completions` と `POST /api/v1/chat` で利用できます。
- 既定モデルは `mlx-community/Qwen3.5-9B-4bit` です。
- 音声入力は `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` で確認済みです。
- 音声はローカルパス、`file://`、`http(s)` URL を使えます。`data_url` には未対応です。
- 動画は `data_url` 未対応です。`http(s)` または `file://` の URL を使ってください。
- 音声と動画の同時入力には未対応です。
- 動画要約で回答が途中で切れる場合は `max_tokens` を増やしてください（目安: `48` 以上）。
- 初回起動時はモデルダウンロードに時間がかかります。
- `mlx-vlm` は Apple Silicon + MLX 環境向けです。
