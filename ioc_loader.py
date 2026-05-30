"""Thread-safe IOC CSV loader using pandas.

CSV files expected:
- data/blacklist_ips.csv: first column = ip
- data/whitelist_ips.csv: first column = ip
- data/blacklist_ports.csv: first column = port

Why pandas here?
- pandas reads the CSV file in one operation instead of us manually looping over rows.
- After loading, we still convert values to Python frozensets because set lookup is very fast
  when workers check every packet.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet

import pandas as pd

LOGGER = logging.getLogger("ids.ioc_loader")

HEADER_WORDS = {"ip", "ips", "port", "ports"}


@dataclass(frozen=True)
class IOCSnapshot:
    blacklist_ips: FrozenSet[str]
    whitelist_ips: FrozenSet[str]
    blacklist_ports: FrozenSet[int]


class IOCCache:
    """Keeps IOC CSV values in memory for fast O(1) lookup.

    We do NOT read CSV files for every packet. That would be slow.
    Instead, we load once, then reload periodically in a background thread.

    The snapshot is protected by a lock because multiple worker threads read it
    while the reload thread may replace it.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._lock = threading.RLock()
        self._snapshot = IOCSnapshot(
            blacklist_ips=frozenset(),
            whitelist_ips=frozenset(),
            blacklist_ports=frozenset(),
        )

    def load_once(self) -> None:
        blacklist_ips = frozenset(
            _read_ip_csv_pandas(self.data_dir / "blacklist_ips.csv")
        )
        whitelist_ips = frozenset(
            _read_ip_csv_pandas(self.data_dir / "whitelist_ips.csv")
        )
        blacklist_ports = frozenset(
            _read_port_csv_pandas(self.data_dir / "blacklist_ports.csv")
        )

        with self._lock:
            self._snapshot = IOCSnapshot(
                blacklist_ips=blacklist_ips,
                whitelist_ips=whitelist_ips,
                blacklist_ports=blacklist_ports,
            )

        LOGGER.info(
            "IOC loaded: %d blacklist IPs, %d whitelist IPs, %d blacklist ports",
            len(blacklist_ips),
            len(whitelist_ips),
            len(blacklist_ports),
        )

    def snapshot(self) -> IOCSnapshot:
        """Return the current IOC sets.

        frozenset is immutable, so workers can safely use the returned object
        without copying it.
        """
        with self._lock:
            return self._snapshot

    def reload_loop(self, stop_event: threading.Event, interval_seconds: int) -> None:
        """Background thread: reload IOC CSVs every N seconds."""
        while not stop_event.is_set():
            try:
                self.load_once()
            except Exception:
                LOGGER.exception("Could not reload IOC CSV files")
            stop_event.wait(interval_seconds)


def _read_first_column_pandas(path: Path) -> pd.Series:
    """Read the first CSV column using pandas and return cleaned string values.

    This supports both styles:

    With header:
        ip
        192.168.10.10

    Without header:
        192.168.10.10
        192.168.10.20

    Lines starting with # are ignored.
    """
    if not path.exists():
        LOGGER.warning("CSV does not exist: %s", path)
        return pd.Series([], dtype="string")

    try:
        # header=None means pandas will not accidentally treat the first IOC value
        # as a header. We remove header words like "ip" and "port" below.
        df = pd.read_csv(
            path,
            comment="#",
            header=None,
            usecols=[0],
            names=["value"],
            dtype="string",
            keep_default_na=False,
            skip_blank_lines=True,
        )
    except pd.errors.EmptyDataError:
        return pd.Series([], dtype="string")

    values = df["value"].astype("string").str.strip()
    values = values[(values != "") & (~values.str.lower().isin(HEADER_WORDS))]
    return values.drop_duplicates()


def _normalize_ip(value: object) -> object:
    """Return normalized IP string, or pandas NA if invalid."""
    try:
        return str(ipaddress.ip_address(str(value)))
    except ValueError:
        return pd.NA


def _read_ip_csv_pandas(path: Path) -> set[str]:
    values = _read_first_column_pandas(path)
    normalized = values.map(_normalize_ip)

    invalid_count = int(normalized.isna().sum())
    if invalid_count:
        sample_invalid = values[normalized.isna()].head(5).to_list()
        LOGGER.warning(
            "Ignored %d invalid IP value(s) in %s. Sample: %s",
            invalid_count,
            path,
            sample_invalid,
        )

    valid_ips = normalized.dropna().astype(str)
    return set(valid_ips.to_numpy())


def _read_port_csv_pandas(path: Path) -> set[int]:
    values = _read_first_column_pandas(path)

    numeric_ports = pd.to_numeric(values, errors="coerce")
    valid_mask = numeric_ports.between(1, 65535, inclusive="both")

    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        sample_invalid = values[~valid_mask].head(5).to_list()
        LOGGER.warning(
            "Ignored %d invalid port value(s) in %s. Sample: %s",
            invalid_count,
            path,
            sample_invalid,
        )

    valid_ports = numeric_ports[valid_mask].dropna().astype("int64").drop_duplicates()
    return set(valid_ports.to_numpy().tolist())
