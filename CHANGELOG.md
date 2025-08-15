# Changelog

## v4.7.4 - 2025-08-15

This release marks a major upgrade to the system's data ingestion, universe selection, and operational robustness. The focus was on replacing static "magic numbers" with adaptive, data-driven logic and ensuring high-quality data feeds the alpha generation process.

### ✨ New Features & Major Enhancements

*   **Adaptive Universe Screening (`universe_screener.py`)**
    *   Replaced static volume/price thresholds with a dynamic, percentile-based ranking system.
    *   Introduced a **Composite Liquidity Score** based on 24h volume, order book depth, bid-ask spread, and price impact to provide a holistic view of market quality.
    *   Selection is now performed on a per-venue basis to handle exchange heterogeneity before being combined.
    *   Added hysteresis to the selection logic to reduce asset churn at the cutoff threshold.

*   **Multi-Venue Data Harvesting (`data_harvester.py`)**
    *   Upgraded the harvester to support multiple exchanges (e.g., Coinbase, Kraken) with a fallback mechanism.
    *   The system now merges a static `symbols.yaml` with a dynamically generated `symbols.generated.yaml` to ensure a complete and accurate map for all venues.

*   **Data Quality Control (QC) Layer (`data_harvester.py`)**
    *   Integrated a new QC pipeline into the harvesting process to enforce data integrity.
    *   **History Gate:** Rejects assets that do not have a minimum required history (e.g., 180 days).
    *   **Gap Gate:** Analyzes and rejects assets with significant data gaps, ensuring cleaner data for feature generation.
    *   Generates a `harvest_qc.json` report for transparent auditing of the harvesting process.

*   **Configuration Enhancements (`configs/config.yaml`)**
    *   Added new configuration sections for `adaptive_gating`, `liquidity_score`, and harvester `quality_control` to make these new features easily configurable.

### 🐛 Bug Fixes

*   **LLM Proposer (`llm/proposer.py`)**:
    *   Removed the `temperature` parameter from OpenAI API calls to resolve compatibility warnings and streamline alpha generation requests.

### ⚙️ Operational Improvements

*   **System Stability (`install_and_launch_v473.sh`)**:
    *   Validated and refined the main launch script for robust, idempotent startup of all system components (`trading_session`, `dashboard_session`, `hyper_lab`) using `tmux`.
