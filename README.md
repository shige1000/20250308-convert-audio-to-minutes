# 音声書き起こしアプリ

このアプリケーションは、音声ファイルをアップロードすると自動的に書き起こしと話者分離を行います。

## 必要条件

- Docker
- Docker Compose
- HuggingFaceのアカウントとアクセストークン
  - [HuggingFace](https://huggingface.co/)でアカウントを作成
  - [Access Tokens](https://huggingface.co/settings/tokens)からトークンを取得
  - pyannote/speaker-diarization-3.0モデルの使用許可を取得

## 使用方法

### Docker Composeを使用する場合（推奨）:

```bash
docker-compose up --build
```

### 従来のDockerコマンドを使用する場合:

1. Dockerイメージのビルド:
```bash
docker build -t audio-transcription .
```

2. アプリケーションの起動:
```bash
docker run -p 8501:8501 audio-transcription
```

### アプリケーションの使用

1. ブラウザで http://localhost:8501 にアクセス

2. HuggingFaceのアクセストークンを入力

3. 音声ファイル（WAVまたはMP3形式）をアップロード

4. 処理が完了すると、話者ごとに分けられた書き起こし結果が表示されます

## 機能

- 音声の自動書き起こし（Whisperモデル使用）
- 話者の自動分離（pyannote.audio使用）
- 日本語を含む多言語対応
- WAVおよびMP3形式のファイルに対応

## 注意事項

- 処理時間は音声ファイルの長さによって変わります
- 話者分離の精度は録音品質や話者の数によって変動する可能性があります

## 開発について

Docker Composeを使用することで、以下のメリットがあります：
- ホストマシンのコードの変更がコンテナに即時反映されます
- アプリケーションの再起動が自動的に行われます
- 環境変数の管理が容易になります
