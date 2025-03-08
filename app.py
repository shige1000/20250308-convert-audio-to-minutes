import streamlit as st
import whisper
import tempfile
import os
from pydub import AudioSegment
from datetime import timedelta
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.signal import wiener


# Streamlitの設定
st.title("音声書き起こしアプリ")
st.write("音声/動画ファイルをアップロードすると、書き起こしを行います。")


def enhance_audio(audio_segment):
    """音声の品質を改善"""
    # 音量の正規化
    audio_segment = audio_segment.normalize()

    # 音声データをnumpy配列に変換
    samples = np.array(audio_segment.get_array_of_samples())

    # Wienerフィルタでノイズ削減
    enhanced_samples = wiener(samples)

    # 音声データを再構築
    enhanced_audio = audio_segment._spawn(enhanced_samples.astype(np.int16))
    return enhanced_audio


def convert_to_wav(input_file, temp_path):
    """動画/音声ファイルをWAVに変換して前処理"""
    audio = AudioSegment.from_file(input_file)

    # 音声の品質改善
    enhanced_audio = enhance_audio(audio)
    enhanced_audio.export(temp_path, format="wav")


def format_time(seconds):
    """秒数を時:分:秒の形式に変換"""
    return str(timedelta(seconds=seconds)).split('.')[0]


def transcribe_segment(segment_path, model):
    """音声セグメントの書き起こし"""
    return model.transcribe(
        segment_path,
        language="ja",
        task="transcribe",
        word_timestamps=True
    )


def process_audio(audio_file):
    """音声処理のメイン関数"""
    # GPUが利用可能な場合は使用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"使用デバイス: {device}")

    # 一時ファイルの作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_path = temp_audio.name
        # 入力ファイルをWAVに変換
        convert_to_wav(audio_file, temp_path)

        try:
            # Whisperモデルの読み込み（mediumモデルを使用）
            model = whisper.load_model("medium", device=device)

            # 音声ファイルを30秒のセグメントに分割
            audio = AudioSegment.from_wav(temp_path)
            segment_length = 30 * 1000  # 30秒
            segments = []

            for i in range(0, len(audio), segment_length):
                segment = audio[i:i + segment_length]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_segment:
                    segment.export(temp_segment.name, format="wav")
                    segments.append(temp_segment.name)

            # 並列処理で書き起こし
            results = []
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(transcribe_segment, segment_path, model)
                    for segment_path in segments
                ]

                # プログレスバーの表示
                progress_bar = st.progress(0)
                for i, future in enumerate(futures):
                    result = future.result()
                    results.append(result)
                    progress_bar.progress((i + 1) / len(futures))

            # 結果の統合
            combined_segments = []
            time_offset = 0
            for result in results:
                for segment in result["segments"]:
                    segment["start"] += time_offset
                    segment["end"] += time_offset
                    combined_segments.append(segment)
                time_offset += 30  # 30秒ずつオフセット

            return {"segments": combined_segments}

        finally:
            # 一時ファイルの削除
            os.unlink(temp_path)
            for segment_path in segments:
                try:
                    os.unlink(segment_path)
                except:
                    pass


# ファイルアップローダーの表示
uploaded_file = st.file_uploader(
    "音声/動画ファイルをアップロードしてください",
    type=['wav', 'mp3', 'mp4', 'm4a']
)


if uploaded_file:
    with st.spinner('書き起こし処理中...'):
        try:
            result = process_audio(uploaded_file)

            # 結果の表示
            st.subheader("書き起こし結果")

            # セグメントごとに表示
            for segment in result["segments"]:
                start_time = format_time(int(segment["start"]))
                end_time = format_time(int(segment["end"]))
                text = segment["text"].strip()

                st.markdown(f"**[{start_time} - {end_time}]** {text}")

            # テキスト全体のダウンロードボタン
            full_text = "\n".join([
                f"[{format_time(int(segment['start']))} - {format_time(int(segment['end']))}] {segment['text'].strip()}"
                for segment in result["segments"]
            ])

            st.download_button(
                label="テキストをダウンロード",
                data=full_text,
                file_name="transcription.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
