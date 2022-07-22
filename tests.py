import config
import torch
import unittest
from models import YOLOv1
from utils import SumSquaredErrorLoss


class TestModel(unittest.TestCase):
    def test_model(self):
        batch_size = 128
        test_model = YOLOv1()
        test_tensor = torch.rand((batch_size, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
        result = test_model.forward(test_tensor)
        self.assertEqual(tuple(result.size()), (128, config.S, config.S, test_model.depth))


class TestLossFunction(unittest.TestCase):
    def test_zeros(self):
        test = torch.zeros((config.S, config.S, 5 * config.B + config.C))
        loss_func = SumSquaredErrorLoss()
        result = loss_func(test, test)
        self.assertEqual(tuple(result.size()), ())
        self.assertEqual(result.item(), 0)


if __name__ == '__main__':
    unittest.main()
