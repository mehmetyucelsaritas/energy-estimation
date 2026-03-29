import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from codecarbon import EnergyCheckpoint, EnergySegment
from codecarbon.core.units import Energy
from codecarbon.emissions_tracker import OfflineEmissionsTracker
from tests.testutils import get_custom_mock_open

empty_conf = "[codecarbon]"


class TestEnergyCheckpointDataclass(unittest.TestCase):
    def test_segment_since(self):
        a = EnergyCheckpoint(
            monotonic_time_s=10.0,
            cpu_energy_kwh=0.1,
            gpu_energy_kwh=0.2,
            ram_energy_kwh=0.05,
            energy_consumed_kwh=0.35,
            water_litres=1.0,
            emissions_kg=0.01,
        )
        b = EnergyCheckpoint(
            monotonic_time_s=20.0,
            cpu_energy_kwh=0.15,
            gpu_energy_kwh=0.25,
            ram_energy_kwh=0.06,
            energy_consumed_kwh=0.46,
            water_litres=1.1,
            emissions_kg=0.02,
        )
        seg = b.segment_since(a)
        self.assertIsInstance(seg, EnergySegment)
        self.assertAlmostEqual(seg.duration_s, 10.0)
        self.assertAlmostEqual(seg.cpu_energy_kwh, 0.05)
        self.assertAlmostEqual(seg.gpu_energy_kwh, 0.05)
        self.assertAlmostEqual(seg.energy_consumed_kwh, 0.11)
        self.assertAlmostEqual(seg.water_litres, 0.1)
        self.assertAlmostEqual(seg.emissions_kg, 0.01)

    def test_segment_since_rejects_inverted_order(self):
        earlier = EnergyCheckpoint(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        later = EnergyCheckpoint(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            later.segment_since(earlier)


class TestTrackerCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        patcher = mock.patch(
            "builtins.open", new_callable=get_custom_mock_open(empty_conf, empty_conf)
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_checkpoint_without_start_returns_none(self):
        tracker = OfflineEmissionsTracker(
            country_iso_code="USA", save_to_file=False, measure_power_secs=60
        )
        self.assertIsNone(tracker.checkpoint())

    def test_checkpoint_measure_false_skips_measure(self):
        tracker = OfflineEmissionsTracker(
            country_iso_code="USA", save_to_file=False, measure_power_secs=60
        )
        tracker.start()
        with mock.patch.object(tracker, "_measure_power_and_energy") as m:
            cp = tracker.checkpoint(measure=False)
        m.assert_not_called()
        self.assertIsNotNone(cp)
        self.assertEqual(cp.energy_consumed_kwh, 0.0)
        tracker.stop()

    def test_checkpoint_segment_reflects_injected_energy(self):
        tracker = OfflineEmissionsTracker(
            country_iso_code="USA", save_to_file=False, measure_power_secs=60
        )
        tracker.start()

        def add_cpu_kwh(amount: float):
            e = Energy.from_energy(kWh=amount)
            tracker._total_energy += e
            tracker._total_cpu_energy += e
            tracker._last_measured_time = time.perf_counter()

        with mock.patch.object(
            tracker, "_measure_power_and_energy", side_effect=lambda: add_cpu_kwh(0.001)
        ):
            c0 = tracker.checkpoint()
        with mock.patch.object(
            tracker, "_measure_power_and_energy", side_effect=lambda: add_cpu_kwh(0.002)
        ):
            c1 = tracker.checkpoint()

        seg = c1.segment_since(c0)
        self.assertAlmostEqual(seg.cpu_energy_kwh, 0.002)
        self.assertAlmostEqual(seg.energy_consumed_kwh, 0.002)
        tracker.stop()
