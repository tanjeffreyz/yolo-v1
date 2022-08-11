import config
import torch
import unittest
from models import YOLOv1
from utils import SumSquaredErrorLoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestModel(unittest.TestCase):
    def test_shape(self):
        batch_size = 128
        test_model = YOLOv1().to(device)
        test_tensor = torch.rand((batch_size, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])).to(device)
        result = test_model.forward(test_tensor)
        self.assertEqual(tuple(result.size()), (128, config.S, config.S, test_model.depth))


class TestLossFunction(unittest.TestCase):
    SHAPE = (config.S, config.S, 5 * config.B + config.C)

    def test_zeros(self):
        test = torch.zeros(TestLossFunction.SHAPE)
        loss_func = SumSquaredErrorLoss()
        result = loss_func(test, test)
        self.assertEqual(tuple(result.size()), ())
        self.assertEqual(0, result.item())

    def test_positives(self):
        test = torch.rand(TestLossFunction.SHAPE)
        loss_func = SumSquaredErrorLoss()
        result = loss_func(test + 1.0, test + 1.0)
        self.assertEqual(tuple(result.size()), ())
        self.assertFalse(torch.isnan(result).item())
        self.assertTrue(result.item() >= 0)

    def test_negatives(self):
        test = torch.rand(TestLossFunction.SHAPE) - 1.0
        loss_func = SumSquaredErrorLoss()
        result = loss_func(test - 1.0, test + 1.0)
        self.assertEqual(tuple(result.size()), ())
        self.assertFalse(torch.isnan(result).item())
        self.assertTrue(result.item() >= 0)

    def test_single_bbox(self):
        truth = torch.zeros(TestLossFunction.SHAPE)
        truth[0, 0, 4] = 1.0        # Bbox confidence
        truth[0, 0, -1] = 1.0       # Class
        pred = torch.zeros(TestLossFunction.SHAPE)
        pred[0, 0, 0:5] = torch.ones(5)
        loss_func = SumSquaredErrorLoss()
        result = loss_func(pred, truth)
        self.assertEqual(tuple(result.size()), ())
        self.assertEqual(21.0, result.item())

    def test_double_bbox(self):
        truth = torch.zeros(TestLossFunction.SHAPE)
        truth[0, 0, 4] = 1.0        # Bbox confidences
        truth[0, 0, 9] = 1.0
        truth[0, 0, -1] = 1.0       # Class
        pred = torch.zeros(TestLossFunction.SHAPE)
        pred[0, 0, 0:10] = torch.ones(10)
        loss_func = SumSquaredErrorLoss()
        result = loss_func(pred, truth)
        self.assertEqual(tuple(result.size()), ())
        self.assertEqual(41.0, result.item())

    def test_noobj(self):
        truth = torch.zeros(TestLossFunction.SHAPE)
        pred = torch.zeros(TestLossFunction.SHAPE)
        pred[0, 0, 0:10] = torch.ones(10)
        loss_func = SumSquaredErrorLoss()
        result = loss_func(pred, truth)
        self.assertEqual(tuple(result.size()), ())
        self.assertEqual(1.0, result.item())


if __name__ == '__main__':
    unittest.main()
