import tempfile
import unittest
from pathlib import Path

from multievolve.cli.propose import _load_mutation_pool
from multievolve.proposers.base_proposers import CombinatorialProposer
from multievolve.utils.data_utils import MutationListFormats


class CombinatorialProposerTests(unittest.TestCase):
    def setUp(self):
        self.sequence = "ACDEFG"
        self.pool = ["A1V", "A1G", "C2S", "D3N", "E4Q"]

    def test_generates_only_requested_loads_and_counts_position_alternatives(self):
        proposer = CombinatorialProposer(
            start_seq=self.sequence,
            mutation_pool=self.pool,
            min_mutations=2,
            max_mutations=3,
            num_seeds=-1,
        )

        self.assertEqual(proposer.candidate_counts(), {2: 9, 3: 7})
        proposals = proposer.propose()
        self.assertEqual(len(proposals), 16)
        self.assertEqual(set(proposals["num_muts"]), {2, 3})
        self.assertFalse(
            any("A1V" in mutations and "A1G" in mutations for mutations in proposals["Mutations"])
        )

    def test_legacy_trust_radius_is_an_alias_for_max_mutations(self):
        proposer = CombinatorialProposer(
            start_seq=self.sequence,
            mutation_pool=self.pool,
            trust_radius=3,
            num_seeds=-1,
        )

        self.assertEqual(proposer.min_mutations, 2)
        self.assertEqual(proposer.max_mutations, 3)
        self.assertEqual(set(proposer.propose()["num_muts"]), {2, 3})

    def test_rejects_invalid_ranges(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            CombinatorialProposer(
                start_seq=self.sequence,
                mutation_pool=self.pool,
                min_mutations=1,
                max_mutations=2,
            )
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            CombinatorialProposer(
                start_seq=self.sequence,
                mutation_pool=self.pool,
                min_mutations=4,
                max_mutations=3,
            )
        with self.assertRaisesRegex(ValueError, "distinct mutation positions"):
            CombinatorialProposer(
                start_seq=self.sequence,
                mutation_pool=self.pool,
                min_mutations=2,
                max_mutations=5,
            )

    def test_direct_proposer_rejects_invalid_substitutions(self):
        invalid_cases = [
            (["A1A", "C2S"], "no-op"),
            (["A1B", "C2S"], "unsupported amino acid"),
            (["A7V", "C2S"], "outside the WT sequence"),
            (["V1A", "C2S"], "does not match"),
        ]
        for pool, message in invalid_cases:
            with self.subTest(pool=pool):
                with self.assertRaisesRegex(ValueError, message):
                    CombinatorialProposer(
                        start_seq=self.sequence,
                        mutation_pool=pool,
                        min_mutations=2,
                        max_mutations=2,
                        num_seeds=-1,
                    )

    def test_direct_proposer_preserves_first_seen_order(self):
        proposer = CombinatorialProposer(
            start_seq=self.sequence,
            mutation_pool=["D3N", "A1V", "C2S", "A1G"],
            min_mutations=2,
            max_mutations=2,
            num_seeds=-1,
        )
        self.assertEqual(list(proposer._mutations_by_position), [3, 1, 2])
        self.assertEqual(proposer._mutations_by_position[1], ["A1V", "A1G"])

    def test_mutation_pool_deduplication_preserves_first_seen_order(self):
        formatted = MutationListFormats(
            ["D3N/A1V", "C2S/D3N", "A1G"],
            self.sequence,
        )
        self.assertEqual(formatted.get_mutation_pool(), ["D3N", "A1V", "C2S", "A1G"])

    def test_cli_pool_validation_normalizes_and_checks_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            pool_path = Path(directory) / "pool.csv"
            pool_path.write_text("a1v\nC2S\n")
            self.assertEqual(_load_mutation_pool(pool_path, self.sequence), ["A1V", "C2S"])

            pool_path.write_text("A1V\nA1V\n")
            with self.assertRaisesRegex(ValueError, "duplicate mutation"):
                _load_mutation_pool(pool_path, self.sequence)

            pool_path.write_text("V1A\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                _load_mutation_pool(pool_path, self.sequence)

            pool_path.write_text("A1B\n")
            with self.assertRaisesRegex(ValueError, "unsupported amino acid"):
                _load_mutation_pool(pool_path, self.sequence)

            pool_path.write_text("A1V\n\nC2S\n")
            with self.assertRaisesRegex(ValueError, "invalid single mutation"):
                _load_mutation_pool(pool_path, self.sequence)


if __name__ == "__main__":
    unittest.main()
