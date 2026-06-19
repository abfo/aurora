import logging
import requests
from settings import settings

log = logging.getLogger("aurora").getChild("analytics")


class Analytics:
    """Posts simple event analytics to an HTTP endpoint when configured."""

    def __init__(self) -> None:
        url = (settings.analytics_url or "").strip()
        source = (settings.analytics_source or "").strip()
        api_key = (settings.analytics_api_key or "").strip()
        self._enabled = bool(url and source and api_key)
        self._url = url
        self._source = source
        self._api_key = api_key

    @property
    def enabled(self) -> bool:
        return self._enabled

    def report_event(self, category: str) -> None:
        """Post an analytics event. Silently does nothing when not configured."""
        if not self._enabled:
            return
        response = None
        try:
            response = requests.post(
                self._url,
                json={
                    "Source": self._source,
                    "Category": category,
                    "ApiKey": self._api_key,
                },
            )
            response.raise_for_status()
        except Exception:
            log.error("Failed to report analytics event: %s", category, exc_info=True)
        finally:
            if response is not None:
                response.close()
