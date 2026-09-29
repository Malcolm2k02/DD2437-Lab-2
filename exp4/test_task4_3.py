"""Run with python -m unittest exp4.test_task4_3."""
import unittest
import numpy as np
from exp4.task4_3 import (ROOT, load_data, train_som, winners,
                         display_positions, within_cell_agreement, neighbourhood_mask)


class VotingSOMTests(unittest.TestCase):
    def test_data_alignment_and_encoding(self):
        votes, labels, names = load_data(ROOT / "data")
        self.assertEqual(votes.shape, (349, 31))
        self.assertEqual(len(names), 349)
        self.assertEqual(names[0], "Skårman Carl-Erik")
        self.assertEqual(labels["party"][0], 1)
        self.assertEqual(labels["gender"][0], 0)
        self.assertEqual(labels["district"][0], 1)
        self.assertTrue(np.any(votes == .5))

    def test_euclidean_winner_not_dot_product(self):
        np.testing.assert_array_equal(winners(np.array([[.1, .1]]),
                                             np.array([[.1, .1], [1., 1.]])), [0])

    def test_planar_neighbours_do_not_wrap(self):
        grid = np.column_stack(np.unravel_index(np.arange(100), (10, 10)))
        np.testing.assert_array_equal(np.flatnonzero(neighbourhood_mask(grid, 0, 1)),
                                      [0, 1, 10, 11])
        np.testing.assert_array_equal(np.flatnonzero(neighbourhood_mask(grid, 99, 0)),
                                      [99])

    def test_training_is_reproducible_and_bounded(self):
        votes = np.array([[0., 0.], [1., 1.]] * 10)
        first = train_som(votes, epochs=20, seed=7)
        second = train_som(votes, epochs=20, seed=7)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((first[0] >= 0) & (first[0] <= 1)))
        self.assertLess(first[2][-1], first[2][0])
        self.assertNotEqual(first[1][0], first[1][1])

    def test_packing_preserves_cells_and_all_members(self):
        bmu = np.array([0] * 40 + [99] * 3)
        pos = display_positions(bmu)
        self.assertEqual(len(np.unique(pos, axis=0)), len(bmu))
        np.testing.assert_array_equal(np.floor(pos[:, 0] + .5), bmu % 10)
        np.testing.assert_array_equal(np.floor(pos[:, 1] + .5), bmu // 10)

    def test_pair_agreement_has_known_answer(self):
        # First cell: 1 of 3 pairs matches. Second cell: its only pair matches.
        self.assertEqual(within_cell_agreement(np.array([0, 0, 0, 1, 1]),
                                               np.array([1, 1, 2, 3, 3])), .5)


if __name__ == "__main__":
    unittest.main()
