#!/usr/bin/env python3
"""Tests for the Ballerina Model Improvement Pipeline."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import shutil
import tempfile

# Mock GPU-dependent modules BEFORE importing run_pipeline
sys.modules["unsloth"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["torch"] = MagicMock()

import run_pipeline
from run_pipeline import ErrorCategory


class TestExtractCode(unittest.TestCase):
    """Test code extraction from model responses."""

    def test_extract_code_ballerina_block(self):
        response = "Here is the code:\n```ballerina\nimport ballerina/io;\n```"
        code = run_pipeline.extract_code(response)
        self.assertEqual(code, "import ballerina/io;")

    def test_extract_code_generic_block(self):
        response = "Here:\n```\nimport ballerina/io;\npublic function main() {}\n```"
        code = run_pipeline.extract_code(response)
        self.assertEqual(code, "import ballerina/io;\npublic function main() {}")

    def test_extract_code_no_block(self):
        response = "import ballerina/io;"
        code = run_pipeline.extract_code(response)
        self.assertEqual(code, "import ballerina/io;")

    def test_extract_code_prefers_ballerina_block(self):
        response = "```ballerina\nreal code\n```\n```python\nwrong code\n```"
        code = run_pipeline.extract_code(response)
        self.assertEqual(code, "real code")


class TestErrorCategorization(unittest.TestCase):
    """Test that errors are correctly categorized."""

    def test_compile_error(self):
        cat = run_pipeline.categorize_error(
            1, "ERROR: compilation error\nundefined variable 'x'", "", ""
        )
        self.assertEqual(cat, ErrorCategory.COMPILE_ERROR)

    def test_runtime_error(self):
        cat = run_pipeline.categorize_error(
            1, "error: {ballerina}json:Error\narray index out of range", "", ""
        )
        self.assertEqual(cat, ErrorCategory.RUNTIME_ERROR)

    def test_timeout(self):
        cat = run_pipeline.categorize_error(-1, "Execution Timed Out", "", "")
        self.assertEqual(cat, ErrorCategory.TIMEOUT)

    def test_wrong_answer(self):
        cat = run_pipeline.categorize_error(0, "", "wrong_output", "expected_output")
        self.assertEqual(cat, ErrorCategory.WRONG_ANSWER)

    def test_success(self):
        cat = run_pipeline.categorize_error(0, "", "expected_output", "expected_output")
        self.assertEqual(cat, ErrorCategory.SUCCESS)

    def test_empty_stdout_with_stderr_is_runtime_error(self):
        """bal run can return 0 but program crashes silently with error in stderr."""
        cat = run_pipeline.categorize_error(
            0, "error: {ballerina}lang.value:ConversionError", "", "expected"
        )
        self.assertEqual(cat, ErrorCategory.RUNTIME_ERROR)

    def test_empty_stdout_with_compile_stderr_is_compile_error(self):
        cat = run_pipeline.categorize_error(
            0, "ERROR: compilation error\nundefined symbol 'foo'", "", "expected"
        )
        self.assertEqual(cat, ErrorCategory.COMPILE_ERROR)

    def test_empty_stdout_no_stderr_is_wrong_answer(self):
        """No stderr means the program just didn't print anything."""
        cat = run_pipeline.categorize_error(0, "", "", "expected")
        self.assertEqual(cat, ErrorCategory.WRONG_ANSWER)


class TestCleanStderr(unittest.TestCase):
    """Test that stderr is cleaned properly."""

    def test_strips_paths(self):
        stderr = "ERROR /Users/foo/bar/tmpdir/main.bal:5:10 undefined variable"
        cleaned = run_pipeline.clean_stderr(stderr)
        self.assertIn("main.bal", cleaned)
        self.assertNotIn("/Users/foo/bar/tmpdir/", cleaned)

    def test_strips_compiling_line(self):
        stderr = "Compiling source\nmain.bal:3:1 missing semicolon"
        cleaned = run_pipeline.clean_stderr(stderr)
        self.assertNotIn("Compiling", cleaned)
        self.assertIn("missing semicolon", cleaned)

    def test_strips_java_stack_traces(self):
        stderr = "error: something\n\tat java.lang.Thread.run(Thread.java:750)\n\tat io.ballerina.runtime.internal.BalRuntime.start(BalRuntime.java:100)"
        cleaned = run_pipeline.clean_stderr(stderr)
        self.assertIn("error: something", cleaned)
        self.assertNotIn("java.lang", cleaned)
        self.assertNotIn("io.ballerina", cleaned)

    def test_keeps_ballerina_stack_traces(self):
        stderr = "error: panic\n\tat module.main(main.bal:10)\n\tat module.foo(utils.bal:5)"
        cleaned = run_pipeline.clean_stderr(stderr)
        self.assertIn("error: panic", cleaned)
        self.assertIn("at module.main(main.bal:10)", cleaned)
        self.assertIn("at module.foo(utils.bal:5)", cleaned)


class TestFeedbackPrompt(unittest.TestCase):
    """Test category-specific feedback prompt construction."""

    def test_compile_error_prompt(self):
        prompt = run_pipeline.build_feedback_prompt(
            "missing semicolon", ErrorCategory.COMPILE_ERROR, 0, 4, ""
        )
        self.assertIn("compile", prompt.lower())
        self.assertIn("syntax", prompt.lower())

    def test_wrong_answer_prompt(self):
        prompt = run_pipeline.build_feedback_prompt(
            "Expected: 5 Got: 3", ErrorCategory.WRONG_ANSWER, 0, 4, ""
        )
        self.assertIn("wrong output", prompt.lower())

    def test_timeout_prompt(self):
        prompt = run_pipeline.build_feedback_prompt(
            "Timed out", ErrorCategory.TIMEOUT, 0, 4, ""
        )
        self.assertIn("time complexity", prompt.lower())

    def test_python_hint_on_last_retry(self):
        prompt = run_pipeline.build_feedback_prompt(
            "wrong", ErrorCategory.WRONG_ANSWER, 3, 4, "print(42)"
        )
        self.assertIn("print(42)", prompt)

    def test_no_python_hint_on_early_retry(self):
        prompt = run_pipeline.build_feedback_prompt(
            "wrong", ErrorCategory.WRONG_ANSWER, 0, 4, "print(42)"
        )
        self.assertNotIn("print(42)", prompt)


class TestEvaluateSolution(unittest.TestCase):
    """Test the evaluation function with mocked Ballerina execution."""

    @patch("run_pipeline.run_ballerina_code")
    def test_all_pass(self, mock_run):
        mock_run.return_value = (0, "expected_output", "")
        test_cases = [{"input": "inp", "output": "expected_output"}]
        success, feedback, cat = run_pipeline.evaluate_solution("code", test_cases)
        self.assertTrue(success)
        self.assertEqual(cat, ErrorCategory.SUCCESS)

    @patch("run_pipeline.run_ballerina_code")
    def test_wrong_answer(self, mock_run):
        mock_run.return_value = (0, "wrong_output", "")
        test_cases = [{"input": "inp", "output": "expected_output"}]
        success, feedback, cat = run_pipeline.evaluate_solution("code", test_cases)
        self.assertFalse(success)
        self.assertEqual(cat, ErrorCategory.WRONG_ANSWER)
        self.assertIn("wrong_output", feedback)

    @patch("run_pipeline.run_ballerina_code")
    def test_compile_error(self, mock_run):
        mock_run.return_value = (1, "", "compilation error: missing semicolon")
        test_cases = [{"input": "inp", "output": "expected_output"}]
        success, feedback, cat = run_pipeline.evaluate_solution("code", test_cases)
        self.assertFalse(success)
        self.assertEqual(cat, ErrorCategory.COMPILE_ERROR)

    @patch("run_pipeline.run_ballerina_code")
    def test_timeout(self, mock_run):
        mock_run.return_value = (-1, "", "Execution Timed Out")
        test_cases = [{"input": "inp", "output": "expected_output"}]
        success, feedback, cat = run_pipeline.evaluate_solution("code", test_cases)
        self.assertFalse(success)
        self.assertEqual(cat, ErrorCategory.TIMEOUT)


class TestProgressTracking(unittest.TestCase):
    """Test progress save/load/resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.progress_file = os.path.join(self.tmpdir, "progress.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_nonexistent(self):
        p = run_pipeline.load_progress(os.path.join(self.tmpdir, "nope.json"))
        self.assertEqual(p, {})

    def test_save_and_load(self):
        data = {"0": {"status": "solved", "attempts": 2}}
        run_pipeline.save_progress(self.progress_file, data)
        loaded = run_pipeline.load_progress(self.progress_file)
        self.assertEqual(loaded, data)

    def test_resume_skips_completed(self):
        data = {"0": {"status": "solved"}, "1": {"status": "failed"}}
        run_pipeline.save_progress(self.progress_file, data)
        loaded = run_pipeline.load_progress(self.progress_file)
        self.assertIn("0", loaded)
        self.assertEqual(loaded["0"]["status"], "solved")


if __name__ == "__main__":
    unittest.main()
