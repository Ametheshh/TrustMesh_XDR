"""Configuration-driven adapter registry."""

from .adapters.ciciot23 import CICIOT23Adapter
from .adapters.ciciomt2024_wifi_mqtt import CICIoMT2024WifiMqttAdapter
from .adapters.ctu_sme_zeek import CTUSMEZeekAdapter
from .adapters.unsw_named import UNSWNamedAdapter
from .adapters.unsw_partitions import UNSWPartitionAdapter

ADAPTERS = {
    "ciciot23": CICIOT23Adapter,
    "ciciomt2024_wifi_mqtt": CICIoMT2024WifiMqttAdapter,
    "unsw_nb15_named": UNSWNamedAdapter,
    "unsw_nb15_partitions": UNSWPartitionAdapter,
    "ctu_sme_zeek": CTUSMEZeekAdapter,
}


def get_adapter(adapter_id: str):
    try:
        return ADAPTERS[adapter_id]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter: {adapter_id}") from exc
