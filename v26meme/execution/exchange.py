import os
from loguru import logger
try:
    from google.cloud import secretmanager
except Exception:
    secretmanager = None
import ccxt

class ExchangeFactory:
    def __init__(self, gcp_project_id: str | None = None):
        self.gcp_project_id = gcp_project_id
        self.secret_client = secretmanager.SecretManagerServiceClient() if (gcp_project_id and secretmanager) else None

    def _get_secret(self, key: str) -> str:
        if self.secret_client and self.gcp_project_id:
            try:
                name = f"projects/{self.gcp_project_id}/secrets/{key}/versions/latest"
                return self.secret_client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")
            except Exception as e:
                logger.error(f"GCP Secret access failed for {key}: {e}")
        return os.environ.get(key, "")

    def get_exchange(self, exchange_id: str):
        if not hasattr(ccxt, exchange_id):
            raise RuntimeError(f"ccxt has no exchange {exchange_id}")
        exchange_class = getattr(ccxt, exchange_id)
        api_key = self._get_secret(f"{exchange_id.upper()}_API_KEY")
        api_secret = self._get_secret(f"{exchange_id.upper()}_API_SECRET")
        cfg = {'apiKey': api_key, 'secret': api_secret} if api_key and api_secret else {}
        ex = exchange_class(cfg)
        try:
            ex.load_markets()
        except Exception as e:
            logger.warning(f"{exchange_id}.load_markets failed: {e}")
        return ex
