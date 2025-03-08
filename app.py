import streamlit as st
import whisper
import tempfile
import os
from pydub import AudioSegment
from datetime import timedelta
import torch
from typing import Dict, List, Any
from pydub.exceptions import CouldntDecodeError


st.title("音声書き起こしアプリ")
st.write("音声/動画ファイルをアップロードすると、書き起こしを行います。")


def convert_to_wav(input_file: Any, temp_path: str) -> None:
    """動画/音声ファイルをWAVに変換して前処理

    Args:
        input_file: 入力ファイル（Streamlitのアップロードファイル）
        temp_path: 一時保存するWAVファイルのパス
    """
    try:
        # 音声ファイルを読み込み
        audio: AudioSegment = AudioSegment.from_file(input_file)

        # Whisperモデルの要件に合わせて音声を変換
        audio = audio.set_channels(1)  # モノラルに変換
        audio = audio.set_frame_rate(16000)  # 16kHzにサンプリングレート変換
        audio = audio.set_sample_width(2)  # 16ビットに設定

        # 音量の正規化
        audio = audio.normalize()

        # WAVファイルとして保存（FFmpegパラメータを明示的に指定）
        audio.export(
            temp_path,
            format="wav",
            parameters=[
                "-ac", "1",  # モノラル
                "-ar", "16000",  # 16kHz
                "-acodec", "pcm_s16le",  # 16ビットPCM
                "-loglevel", "error"  # FFmpegのログレベルを制限
            ]
        )
    except Exception as e:
        st.error(f"音声変換中にエラーが発生しました: {str(e)}")
        raise


def format_time(seconds: int) -> str:
    """秒数を時:分:秒の形式に変換

    Args:
        seconds: 変換する秒数

    Returns:
        str: HH:MM:SS形式の文字列
    """
    return str(timedelta(seconds=seconds)).split('.')[0]


def process_audio(audio_file: Any) -> Dict[str, List[Dict[str, Any]]]:
    """音声処理のメイン関数

    Args:
        audio_file: Streamlitでアップロードされた音声ファイル

    Returns:
        Dict[str, List[Dict[str, Any]]]: 書き起こし結果のセグメントリスト

    Raises:
        Exception: 音声処理中に発生したエラー
    """
    # GPUが利用可能な場合は使用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"使用デバイス: {device}")

    # 一時ファイルの作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_path: str = temp_audio.name
        try:
            # 入力ファイルをWAVに変換
            try:
                convert_to_wav(audio_file, temp_path)
            except CouldntDecodeError:
                raise Exception("ファイルの形式が正しくないか、破損している可能性があります。")
            except Exception as e:
                raise Exception(f"音声ファイルの変換中にエラーが発生しました: {str(e)}")

            # Whisperモデルの読み込み
            try:
                # tiny (39M パラメータ)
                # base (74M パラメータ)
                # small (244M パラメータ)
                # medium (769M パラメータ)
                # large (1550M パラメータ)
                model: whisper.Whisper = whisper.load_model("tiny", device=device)
            except Exception as e:
                raise Exception(f"Whisperモデルの読み込み中にエラーが発生しました: {str(e)}")

            # 音声ファイルを直接Whisperで処理
            try:
                result = model.transcribe(
                    temp_path,
                    language="ja",
                    task="transcribe",
                    word_timestamps=True
                )
                return {"segments": result["segments"]}

            except Exception as e:
                raise Exception(f"音声処理中にエラーが発生しました: {str(e)}")

        finally:
            # 一時ファイルの削除
            try:
                os.unlink(temp_path)
            except Exception as e:
                st.warning(f"一時ファイルの削除中にエラーが発生しました: {str(e)}")


# ファイルアップローダーの表示
uploaded_file = st.file_uploader(
    "音声/動画ファイルをアップロードしてください",
    type=['wav', 'mp3', 'mp4', 'm4a']
)


if uploaded_file:
    with st.spinner('書き起こし処理中...'):
        try:
            result: Dict[str, List[Dict[str, Any]]] = process_audio(uploaded_file)

            # 結果の表示
            st.subheader("書き起こし結果")

            # セグメントごとに表示
            for segment in result["segments"]:
                start_time: str = format_time(int(segment["start"]))
                end_time: str = format_time(int(segment["end"]))
                text: str = segment["text"].strip()

                st.markdown(f"**[{start_time} - {end_time}]** {text}")

            # テキスト全体のダウンロードボタン
            full_text: str = "\n".join([
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
