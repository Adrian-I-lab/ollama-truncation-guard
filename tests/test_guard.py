import unittest

import guard


class TruncatedCases(unittest.TestCase):
    def test_137k_sent_65538_seen(self):
        reasons = guard.check(65538, 137000, num_ctx=131072)
        self.assertGreaterEqual(len(reasons), 2)
        self.assertTrue(any("half-window" in r for r in reasons))

    def test_168k_sent_65538_seen(self):
        reasons = guard.check(65538, 168000, num_ctx=131072)
        self.assertGreaterEqual(len(reasons), 2)

    def test_reasons_carry_the_numbers(self):
        text = " ".join(guard.check(65538, 137000, num_ctx=131072))
        for n in ("65538", "137000", "131072"):
            self.assertIn(n, text)

    def test_half_window_fires_without_an_estimate(self):
        reasons = guard.check(65538, None, num_ctx=131072)
        self.assertEqual(len(reasons), 1)
        self.assertIn("half-window", reasons[0])

    def test_missing_count(self):
        reasons = guard.check(None, 1000, num_ctx=2048)
        self.assertEqual(len(reasons), 1)
        self.assertIn("missing prompt_eval_count", reasons[0])

    def test_count_exceeds_window(self):
        reasons = guard.check(3000, 2900, num_ctx=2048)
        self.assertTrue(any("exceeds num_ctx" in r for r in reasons))


class HonestCases(unittest.TestCase):
    def test_fits_in_window(self):
        self.assertEqual(guard.check(90416, 95000, num_ctx=131072), [])

    def test_no_window_known(self):
        self.assertEqual(guard.check(1745, 1800, num_ctx=None), [])

    def test_honest_half_window_count_is_not_flagged(self):
        # A prompt that really is about half the window, and was expected to be.
        self.assertEqual(guard.check(65538, 66000, num_ctx=131072), [])


class Controls(unittest.TestCase):
    def test_disabled_checks_let_truncation_through(self):
        # Same truncated numbers, checks switched off: proves the checks decide.
        self.assertEqual(guard.check(65538, 137000, num_ctx=None, tolerance=0), [])


class Estimates(unittest.TestCase):
    def test_lower_bound_on_6000_chars(self):
        self.assertEqual(guard.estimate_tokens("x" * 6000), 1000)

    def test_custom_ratio(self):
        self.assertEqual(guard.estimate_tokens("x" * 6000, chars_per_token=4.0), 1500)

    def test_prompt_text_generate(self):
        body = {"system": "sys ", "prompt": "hello"}
        self.assertEqual(guard.prompt_text("/api/generate", body), "sys hello")
        self.assertEqual(guard.prompt_text("/api/generate", {}), "")

    def test_prompt_text_chat(self):
        body = {"messages": [{"role": "system", "content": "a"}, {"role": "user", "content": "bc"},
                             {"role": "user"}]}
        self.assertEqual(guard.prompt_text("/api/chat", body), "abc")

    def test_preflight(self):
        self.assertIsNotNone(guard.preflight(2500, 2048))
        self.assertIsNone(guard.preflight(1000, 2048))
        self.assertIsNone(guard.preflight(2500, None))

    def test_truncation_error_is_runtime_error(self):
        self.assertTrue(issubclass(guard.TruncationError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
