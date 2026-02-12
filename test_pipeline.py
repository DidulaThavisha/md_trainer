
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import shutil

# Mock unsloth and vllm BEFORE importing run_pipeline
sys.modules["unsloth"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["torch"] = MagicMock()

import run_pipeline

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create dummy log/notes dirs
        os.makedirs("test_log", exist_ok=True)
        os.makedirs("test_notes", exist_ok=True)
        run_pipeline.LOG_DIR = "test_log"
        run_pipeline.NOTES_DIR = "test_notes"

    def tearDown(self):
        # Cleanup
        if os.path.exists("test_log"):
            shutil.rmtree("test_log")
        if os.path.exists("test_notes"):
            shutil.rmtree("test_notes")

    @patch("run_pipeline.run_ballerina_code")
    def test_evaluate_solution_success(self, mock_run):
        # Mock successful run
        mock_run.return_value = (0, "expected_output", "")
        
        test_cases = [{"input": "inp", "output": "expected_output"}]
        success, feedback = run_pipeline.evaluate_solution("code", test_cases)
        
        self.assertTrue(success)
        self.assertEqual(feedback, "Success")

    @patch("run_pipeline.run_ballerina_code")
    def test_evaluate_solution_fail_output(self, mock_run):
        # Mock wrong answer
        mock_run.return_value = (0, "wrong_output", "")
        
        test_cases = [{"input": "inp", "output": "expected_output"}]
        success, feedback = run_pipeline.evaluate_solution("code", test_cases)
        
        self.assertFalse(success)
        self.assertIn("Wrong Answer", feedback)
        self.assertIn("wrong_output", feedback)

    @patch("run_pipeline.run_ballerina_code")
    def test_evaluate_solution_compile_error(self, mock_run):
        # Mock compile error
        mock_run.return_value = (1, "", "Syntax Error")
        
        test_cases = [{"input": "inp", "output": "expected_output"}]
        success, feedback = run_pipeline.evaluate_solution("code", test_cases)
        
        self.assertFalse(success)
        self.assertIn("Compile/Runtime Error", feedback)
        self.assertIn("Syntax Error", feedback)

    def test_extract_code(self):
        response = "Here is the code:\n```ballerina\nimport ballerina/io;\n```"
        code = run_pipeline.extract_code(response)
        self.assertEqual(code, "import ballerina/io;")

        response_no_block = "import ballerina/io;"
        code = run_pipeline.extract_code(response_no_block)
        self.assertEqual(code, "import ballerina/io;")

if __name__ == '__main__':
    unittest.main()
