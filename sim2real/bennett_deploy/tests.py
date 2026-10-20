from __future__ import annotations

import ast
import csv
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from .async_csv import AsyncCsvLogger, CSV_HEADER
from .contract import DeploymentContract, JOINT_ORDER
from .gait import crawl_phase_terms
from .policy import ObservationBuilder, PolicyPipeline
from .runtime import DEFAULT_POLICY, DeploymentRunner, ImuTerms, StartupPhase, build_parser


class DeploymentCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = DeploymentContract.load(DEFAULT_POLICY)

    def test_contract_matches_verified_run(self) -> None:
        contract = self.contract
        self.assertEqual(contract.joint_order, JOINT_ORDER)
        self.assertAlmostEqual(contract.policy_rate_hz, 50.0)
        self.assertAlmostEqual(contract.action_scale_rad, 0.2)
        self.assertAlmostEqual(contract.action_clip, 3.5)
        self.assertEqual(contract.train_default_rad, (0.08, -0.16, -0.08, -0.16, 0.08, -0.16, -0.08, -0.16))
        self.assertAlmostEqual(contract.stiffness, 28.0)
        self.assertAlmostEqual(contract.damping, 2.0)
        self.assertAlmostEqual(contract.gait_frequency_hz, 0.55)
        self.assertAlmostEqual(contract.gait_duty_factor, 0.78)
        self.assertAlmostEqual(contract.gait_swing_height, 0.065)

    def test_gait_terms_match_verified_legacy_formula(self) -> None:
        for elapsed, command in ((0.0, (0.0, 0.0, 0.0)), (0.37, (0.08, 0.0, 0.0)), (1.91, (0.0, -0.1, 0.2))):
            actual = crawl_phase_terms(elapsed, 0.55, 0.78, 0.065, command, 0.025)
            moving = math.sqrt(sum(value * value for value in command)) >= 0.025
            if not moving:
                expected_phase = 0.0
                expected_contacts = (1.0, 1.0, 1.0, 1.0)
            else:
                expected_phase = (elapsed * 0.55) % 1.0
                expected_contacts = tuple(
                    0.0 if (expected_phase - offset) % 1.0 < 0.22 else 1.0
                    for offset in (0.0, 0.5, 0.75, 0.25)
                )
            self.assertAlmostEqual(actual.phase, expected_phase)
            self.assertEqual(actual.desired_contacts, expected_contacts)

    def test_observation_is_exactly_50_values(self) -> None:
        import torch

        builder = ObservationBuilder(self.contract, torch)
        obs, _ = builder.build(
            base_ang_vel=(0.1, 0.2, 0.3),
            projected_gravity=(0.0, 0.0, -1.0),
            velocity_command=(0.08, 0.0, 0.0),
            joint_pos_rel=(0.0,) * 8,
            joint_vel_rel=(0.0,) * 8,
            last_action=torch.zeros(8),
            phase_elapsed_s=0.4,
        )
        self.assertEqual(tuple(obs.shape), (1, 50))
        expected = [0.1, 0.2, 0.3, 0.0, 0.0, -1.0, 0.08, 0.0, 0.0]
        for actual, wanted in zip(obs[0, :9].tolist(), expected):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_policy_preflight(self) -> None:
        pipeline = PolicyPipeline(self.contract, output_scale=1.0, target_rate_limit_deg_s=60000.0)
        result = pipeline.preflight()
        self.assertLess(result["clip_fraction"], 0.05)
        self.assertTrue(math.isfinite(result["raw_abs_max"]))

    def test_extracted_motor_classes_are_ast_identical(self) -> None:
        package_dir = Path(__file__).resolve().parent
        legacy_path = package_dir.parent / "quad_leg_go2-10.py"
        if not legacy_path.is_file():
            self.skipTest(f"legacy script not present: {legacy_path.name}")
        extracted_path = package_dir / "dm_can.py"

        def classes(path: Path):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.ClassDef)}

        legacy = classes(legacy_path)
        extracted = classes(extracted_path)
        self.assertEqual(extracted["Motor"], legacy["Motor"])
        self.assertEqual(extracted["Motor_Control"], legacy["Motor_Control"])

    def test_async_csv_schema_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            logger = AsyncCsvLogger(path)
            logger.submit([tuple(range(len(CSV_HEADER)))])
            diagnostics = logger.close()
            self.assertEqual(diagnostics["written_rows"], 1)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(tuple(rows[0]), CSV_HEADER)
            self.assertEqual(len(rows[1]), len(CSV_HEADER))

    def test_first_position_command_is_exact_startup_pose(self) -> None:
        args = build_parser().parse_args(["--check_only"])
        runner = DeploymentRunner(args, self.contract, threading.Event())
        runner.motor_zero = dict.fromkeys(self.contract.joint_order, 0.0)
        positions = (0.11, -0.22, -0.13, 0.24, 0.15, -0.26, -0.17, 0.28)
        snapshot = {
            name: SimpleNamespace(position=positions[index], velocity=0.0)
            for index, name in enumerate(self.contract.joint_order)
        }
        startup_sim, _, _ = runner._feedback_to_policy(snapshot, runner.motor_zero)
        targets = runner._motor_targets(startup_sim)
        for index, name in enumerate(self.contract.joint_order):
            self.assertAlmostEqual(targets[name], positions[index], places=12)

    def test_software_effort_limit_matches_mit_pd_equation(self) -> None:
        args = build_parser().parse_args(["--check_only", "--software_effort_limit_nm", "8"])
        runner = DeploymentRunner(args, self.contract, threading.Event())
        runner.active_names = tuple(self.contract.joint_order)
        targets = dict.fromkeys(self.contract.joint_order, 1.0)
        feedback = {
            name: SimpleNamespace(position=0.0, velocity=0.5)
            for name in self.contract.joint_order
        }
        limited = runner._software_effort_limited_targets(targets, feedback)
        for name in self.contract.joint_order:
            commanded_tau = runner.kp * (limited[name] - feedback[name].position) - runner.kd * feedback[name].velocity
            self.assertAlmostEqual(commanded_tau, 8.0, places=12)

    def test_runtime_log_rows_match_schema(self) -> None:
        args = build_parser().parse_args(["--check_only"])
        runner = DeploymentRunner(args, self.contract, threading.Event())

        class Capture:
            rows = None

            def submit(self, rows):
                self.rows = tuple(rows)

        capture = Capture()
        runner.logger = capture
        torch = runner.pipeline.torch
        zeros = torch.zeros(8)
        feedback = {
            name: SimpleNamespace(position=0.0, velocity=0.0, torque=0.0, error=0, feedback_age_s=0.0)
            for name in self.contract.joint_order
        }
        gait = crawl_phase_terms(0.0, 0.55, 0.78, 0.065, (0.0, 0.0, 0.0), 0.025)
        imu = ImuTerms((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0),
                       (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.001, 0.001)
        runner._log_rows(
            now=1.0, elapsed_s=0.0, loop_count=1, policy_step_count=0,
            startup_phase=StartupPhase.RAMP_DEFAULT, diag_stage="manual", gait=gait,
            raw_action=zeros, action=zeros, desired_offset=zeros, applied_offset=zeros,
            target_sim=runner.pipeline.train_default, rel_pos=(0.0,) * 8, joint_vel_obs=(0.0,) * 8,
            q_policy=dict.fromkeys(self.contract.joint_order, 0.0),
            q_cmd=dict.fromkeys(self.contract.joint_order, 0.0), feedback=feedback, imu_terms=imu,
        )
        self.assertEqual(len(capture.rows), 8)
        self.assertTrue(all(len(row) == len(CSV_HEADER) for row in capture.rows))


if __name__ == "__main__":
    unittest.main()


class MotorWatchdogTests(unittest.TestCase):
    class FakeControl:
        def __init__(self, initial=0, stuck=False):
            self.regs = {1: initial, 2: initial}
            self.stuck = stuck
            self.mit_frames = 0
            self.disabled = 0

        def read_motor_param(self, motor, rid, timeout=1.0):
            return self.regs[motor]

        def write_motor_param(self, motor, rid, data):
            if not self.stuck:
                self.regs[motor] = int.from_bytes(data, "little")

        def control_mit(self, motor, kp, kd, q, dq, tau):
            self.mit_frames += 1

        def disable_all(self):
            self.disabled += 1

        def close(self):
            pass

    def make_bus(self, **kwargs):
        from .dm_can import Go2MotorBus

        bus = Go2MotorBus.__new__(Go2MotorBus)
        bus.control = self.FakeControl(**kwargs)
        bus.motors = {"a": 1, "b": 2}
        return bus

    def test_arm_verify_and_restore(self) -> None:
        bus = self.make_bus(initial=0)
        bus.arm_watchdog(300.0, 50.0)
        self.assertEqual(bus.control.regs, {1: 6000, 2: 6000})
        self.assertGreater(bus.control.mit_frames, 0)
        bus.close()
        self.assertEqual(bus.control.regs, {1: 0, 2: 0})
        self.assertGreaterEqual(bus.control.disabled, 1)

    def test_readback_mismatch_raises_and_close_restores(self) -> None:
        bus = self.make_bus(initial=7, stuck=True)
        with self.assertRaises(RuntimeError):
            bus.arm_watchdog(300.0, 50.0)
        bus.close()
        self.assertEqual(bus.control.regs, {1: 7, 2: 7})

    def test_too_short_timeout_rejected(self) -> None:
        bus = self.make_bus()
        with self.assertRaises(ValueError):
            bus.arm_watchdog(20.0, 50.0)

    def test_parser_default(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.motor_watchdog_ms, 300.0)


class PosixKeyParserTests(unittest.TestCase):
    def test_plain_keys_and_arm(self) -> None:
        from .input import parse_posix_keys

        self.assertEqual(parse_posix_keys("ws a\n"), [("w", False), ("s", False), (" ", False), ("a", False), ("\r", False)])

    def test_page_keys_home_and_arrows(self) -> None:
        from .input import parse_posix_keys

        self.assertEqual(parse_posix_keys("\x1b[5~\x1b[6~\x1b[H\x1b[1~"), [("I", True), ("Q", True), ("G", True), ("G", True)])
        self.assertEqual(parse_posix_keys("\x1b[A\x1b[B\x1bOC"), [])

    def test_lone_escape_is_shutdown_and_partial_sequence_is_ignored(self) -> None:
        from .input import parse_posix_keys

        self.assertEqual(parse_posix_keys("\x1b"), [("\x1b", False)])
        self.assertEqual(parse_posix_keys("\x1b["), [])
