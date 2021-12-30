"""Temp ball detection test."""
import torch

from models.TTNet import BallDetection


def normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize a tensor image with mean and standard deviation."""
    mean = torch.repeat_interleave(torch.tensor(mean).view(1, 3, 1, 1), repeats=9, dim=1)
    std = torch.repeat_interleave(torch.tensor(std).view(1, 3, 1, 1), repeats=9, dim=1)
    return (x / 255. - mean) / std


def main():
    """Run the ball detection model."""
    for i in range(10):
        ball_detection = BallDetection(9, 0.5)
        normalized_images = normalize(torch.randn(1, 27, 128, 320))
        pred_ball_global, global_features, out_block2, out_block3, out_block4, out_block5 = ball_detection(normalized_images)
        print(i, pred_ball_global)


if __name__ == "__main__":
    main()