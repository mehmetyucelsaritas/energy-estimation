"""
The Carbon Tracker module. The following objects/decorators belong to the Public API
"""

from ._version import __version__  # noqa
from .emissions_tracker import (
    EmissionsTracker,
    OfflineEmissionsTracker,
    track_emissions,
)
from .output_methods.emissions_data import EnergyCheckpoint, EnergySegment

__all__ = [
    "EmissionsTracker",
    "OfflineEmissionsTracker",
    "track_emissions",
    "EnergyCheckpoint",
    "EnergySegment",
]
__app_name__ = "codecarbon"
