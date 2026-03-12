from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from trust_bench.cli import cli


class TestCLI:
    def test_cli_group_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "survey" in result.output
        assert "report" in result.output
        assert "feature" in result.output

    @patch("trust_bench.cli.get_model")
    @patch("trust_bench.cli.get_probe")
    @patch("trust_bench.cli.load_config")
    @patch("trust_bench.cli.save_result")
    @patch("trust_bench.cli.generate_report")
    def test_run_command(self, mock_report, mock_save, mock_config, mock_probe, mock_model):
        mock_config.return_value = {
            "probe": "feature_survey",
            "model": "llama-3.1-8b",
            "layer": 16,
            "prompts": [],
        }
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_probe_instance = MagicMock()
        mock_probe.return_value = mock_probe_instance
        mock_probe_instance.run.return_value = MagicMock()

        runner = CliRunner()
        runner.invoke(cli, ["run", "fake_config.yaml"])

        mock_config.assert_called_once_with("fake_config.yaml")
        mock_model.assert_called_once_with("llama-3.1-8b")
        mock_model_instance.load.assert_called_once()
        mock_probe_instance.validate_config.assert_called_once()
        mock_probe_instance.run.assert_called_once()

    def test_run_with_max_prompts(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert "--max-prompts" in result.output
        assert "--device" in result.output
        assert "--cache-dir" in result.output
