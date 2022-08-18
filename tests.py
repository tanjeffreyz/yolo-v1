import config
import torch
import unittest
import utils
from models import YOLOv1, YOLOv1ResNet
from loss import SumSquaredErrorLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestScratchModel(unittest.TestCase):
    def test_shape(self):
        batch_size = 64
        test_model = YOLOv1().to(device)
        test_tensor = torch.rand((batch_size, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])).to(device)
        result = test_model.forward(test_tensor)
        self.assertEqual(tuple(result.size()), (128, config.S, config.S, test_model.depth))


class TestTransferModels(unittest.TestCase):
    def test_shape(self):
        batch_size = 64
        test_model = YOLOv1ResNet().to(device)
        test_tensor = torch.rand((batch_size, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])).to(device)
        result = test_model.forward(test_tensor)
        self.assertEqual(tuple(result.size()), (batch_size, config.S, config.S, test_model.depth))


class TestLossFunction(unittest.TestCase):
    SHAPE = (config.BATCH_SIZE, config.S, config.S, 5 * config.B + config.C)

    def test_small_positive_iou(self):
        a = torch.zeros((1, 1, 1, TestLossFunction.SHAPE[-1]))
        a[0, 0, 0, config.C:config.C+5] = torch.tensor([1, 1, 1, 1, 1])
        a[0, 0, 0, config.C+5:config.C+10] = torch.tensor([0.5, 0.5, 1, 1, 1])
        b = torch.zeros((1, 1, 1, TestLossFunction.SHAPE[-1]))
        b[0, 0, 0, config.C:config.C+5] = torch.tensor([0.5, 0.5, 1, 1, 1])
        print(utils.get_iou(a, b))

    def test_small_negative_iou(self):
        test = torch.zeros((1, 1, 1, TestLossFunction.SHAPE[-1]))
        test[0, 0, 0, 0:5] = torch.tensor([0, 0, 1, 1, 1])
        print(utils.get_iou(test, test))

    def test_bbox_to_coords_size(self):
        test = torch.rand(TestLossFunction.SHAPE)
        result = utils.bbox_to_coords(test)
        self.assertEqual(result[0].size(), (config.BATCH_SIZE, config.S, config.S, config.B, 2))
        self.assertEqual(result[1].size(), (config.BATCH_SIZE, config.S, config.S, config.B, 2))

    def test_get_iou_size(self):
        test = torch.rand(TestLossFunction.SHAPE)
        result = utils.get_iou(test, test)
        self.assertEqual(result.size(), (config.BATCH_SIZE, config.S, config.S, config.B, config.B))

    def test_torch_max(self):
        test = torch.rand((4, 2, 2))
        print(test)
        # print(torch.max(test, dim=0)[0])
        # print(torch.max(test, dim=1))
        # print(torch.argmax(test, dim=-2).size())
        print(torch.max(test, dim=-2)[0].size())
        print(torch.argmax(torch.max(test, dim=-2)[0], dim=-1, keepdim=True).size())
        print(torch.zeros((4, 2)).scatter_(-1, torch.argmax(torch.max(test, dim=-2)[0], dim=-1, keepdim=True), value=1))

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
        truth[0, 0, 0, 4] = 1.0        # Bbox confidence
        truth[0, 0, 0, -1] = 1.0       # Class
        pred = torch.zeros(TestLossFunction.SHAPE)
        pred[0, 0, 0, 0:5] = torch.ones(5)
        loss_func = SumSquaredErrorLoss()
        result = loss_func(pred, truth)
        self.assertEqual(tuple(result.size()), ())
        self.assertEqual(21.0, result.item())

    def test_double_bbox(self):
        truth = torch.zeros(TestLossFunction.SHAPE)
        truth[0, 0, 0, 4] = 1.0        # Bbox confidences
        truth[0, 0, 0, 9] = 1.0
        truth[0, 0, 0, -1] = 1.0       # Class
        pred = torch.zeros(TestLossFunction.SHAPE)
        pred[0, 0, 0, 0:10] = torch.ones(10)
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
