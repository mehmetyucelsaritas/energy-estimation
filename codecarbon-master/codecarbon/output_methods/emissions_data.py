import json
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class EmissionsData:
    """
    Output object containing run data
    """

    timestamp: str
    project_name: str
    run_id: str
    experiment_id: str
    duration: float
    emissions: float
    emissions_rate: float
    cpu_power: float
    gpu_power: float
    ram_power: float
    cpu_energy: float
    gpu_energy: float
    ram_energy: float
    energy_consumed: float
    water_consumed: float
    country_name: str
    country_iso_code: str
    region: str
    cloud_provider: str
    cloud_region: str
    os: str
    python_version: str
    codecarbon_version: str
    cpu_count: float
    cpu_model: str
    gpu_count: float
    gpu_model: str
    longitude: float
    latitude: float
    ram_total_size: float
    tracking_mode: str
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    ram_utilization_percent: float = 0.0
    ram_used_gb: float = 0.0
    on_cloud: str = "N"
    pue: float = 1
    wue: float = 0

    @property
    def values(self) -> OrderedDict:
        return OrderedDict(self.__dict__.items())

    def compute_delta_emission(self, previous_emission):
        delta_duration = self.duration - previous_emission.duration
        self.duration = delta_duration
        delta_emissions = self.emissions - previous_emission.emissions
        self.emissions = delta_emissions
        self.cpu_energy -= previous_emission.cpu_energy
        self.gpu_energy -= previous_emission.gpu_energy
        self.ram_energy -= previous_emission.ram_energy
        self.energy_consumed -= previous_emission.energy_consumed
        self.water_consumed -= previous_emission.water_consumed
        if delta_duration > 0:
            # emissions_rate in g/s : delta_emissions in kg.CO2 / delta_duration in s
            self.emissions_rate = delta_emissions / delta_duration
        else:
            self.emissions_rate = 0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


@dataclass(frozen=True)
class EnergyCheckpoint:
    """
    Immutable snapshot of cumulative energy, water, and CO2eq at one instant.

    Produced by :meth:`codecarbon.emissions_tracker.BaseEmissionsTracker.checkpoint`.
    Use :meth:`segment_since` to obtain the delta between two checkpoints (e.g. one
    model in a batch) without stopping the tracker or using task mode.
    """

    monotonic_time_s: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    ram_energy_kwh: float
    energy_consumed_kwh: float
    water_litres: float
    emissions_kg: float

    def segment_since(self, earlier: "EnergyCheckpoint") -> "EnergySegment":
        """
        Energy, water, and emissions accumulated between ``earlier`` and this checkpoint.

        ``monotonic_time_s`` on each checkpoint must be from the same ``perf_counter``
        clock (as returned by the tracker). ``earlier`` must be at or before this
        checkpoint in time.
        """
        if earlier.monotonic_time_s > self.monotonic_time_s:
            raise ValueError(
                "earlier.monotonic_time_s must be <= self.monotonic_time_s "
                "(checkpoints must be taken in order)."
            )
        return EnergySegment(
            duration_s=self.monotonic_time_s - earlier.monotonic_time_s,
            cpu_energy_kwh=self.cpu_energy_kwh - earlier.cpu_energy_kwh,
            gpu_energy_kwh=self.gpu_energy_kwh - earlier.gpu_energy_kwh,
            ram_energy_kwh=self.ram_energy_kwh - earlier.ram_energy_kwh,
            energy_consumed_kwh=self.energy_consumed_kwh
            - earlier.energy_consumed_kwh,
            water_litres=self.water_litres - earlier.water_litres,
            emissions_kg=self.emissions_kg - earlier.emissions_kg,
        )


@dataclass(frozen=True)
class EnergySegment:
    """
    Delta between two :class:`EnergyCheckpoint` snapshots.

    Energy fields are exact differences of cumulative tracker totals. ``duration_s``
    is the difference of ``monotonic_time_s`` at snapshot time (after any
    ``measure=True`` work in :meth:`checkpoint`), so it includes checkpoint overhead
    on the second call—not a substitute for timing your model with ``perf_counter``
    if you need pure code duration.
    """

    duration_s: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    ram_energy_kwh: float
    energy_consumed_kwh: float
    water_litres: float
    emissions_kg: float


@dataclass
class TaskEmissionsData:
    task_name: str
    timestamp: str
    project_name: str
    run_id: str
    duration: float
    emissions: float
    emissions_rate: float
    cpu_power: float
    gpu_power: float
    ram_power: float
    cpu_energy: float
    gpu_energy: float
    ram_energy: float
    energy_consumed: float
    water_consumed: float
    country_name: str
    country_iso_code: str
    region: str
    cloud_provider: str
    cloud_region: str
    os: str
    python_version: str
    codecarbon_version: str
    cpu_count: float
    cpu_model: str
    gpu_count: float
    gpu_model: str
    longitude: float
    latitude: float
    ram_total_size: float
    tracking_mode: str
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    ram_utilization_percent: float = 0.0
    ram_used_gb: float = 0.0
    on_cloud: str = "N"

    @property
    def values(self) -> OrderedDict:
        return OrderedDict(self.__dict__.items())
