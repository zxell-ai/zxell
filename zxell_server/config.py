from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """サーバ設定。環境変数（ZXELL_ プレフィックス）で渡すのが正式手順。
    開発時は .env ファイルでも可（書式は .env.example を参照）。

    デフォルトのない項目（db_url, admin_api_key）は未設定だと起動時にエラーになる。
    認証情報はこのファイルには書かないこと。
    """

    model_config = SettingsConfigDict(env_prefix="ZXELL_", env_file=".env", extra="ignore")

    # PostgreSQL 接続 URL（必須）
    db_url: str

    # 管理系 API（タスク投入・重み登録）用キー（必須）
    admin_api_key: str

    # シャード・重み・成果物の保存先ディレクトリ（S3 互換ストレージへの移行は将来課題）
    storage_dir: Path = Path("storage")

    # タスクリースの期限（秒）。期限切れは自動で pending に戻る
    lease_seconds: int = 3600


settings = Settings()
