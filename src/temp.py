"""Temp ball detection test."""
import torch

from models.TTNet import BallDetection


def normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize a tensor image with mean and standard deviation."""
    mean = torch.repeat_interleave(torch.tensor(mean).view(1, 3, 1, 1), repeats=9, dim=1)
    std = torch.repeat_interleave(torch.tensor(std).view(1, 3, 1, 1), repeats=9, dim=1)
    return (x / 255. - mean) / std


def load_pretrained_model(model, pretrained_path):
    """Load weights from the pretrained model"""
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    pretrained_dict = checkpoint['state_dict']
    prefix = "model.ball_global_stage."
    pretrained_dict = {k.replace(prefix, ""): v for k, v in pretrained_dict.items() if k.startswith(prefix)}
    print(pretrained_dict.keys())

    model_state_dict = model.state_dict()
    print(model_state_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
    print(pretrained_dict.keys())
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

    return model


def infer(model, num=10):
    """Perform inference."""
    with torch.no_grad():
        last_output = None
        for i in range(num):
            normalized_images = normalize(torch.randn(1, 27, 128, 320))
            pred_ball_global, global_features, out_block2, out_block3, out_block4, out_block5 = model(normalized_images)
            print(i, torch.sum(pred_ball_global), torch.mean(pred_ball_global))
            if last_output is not None:
                print(torch.all(torch.eq(last_output, pred_ball_global)))
            last_output = pred_ball_global


def main(pretrained_path):
    """Run the ball detection model."""
    model = BallDetection(9, 0.5)
    model.eval()

    infer(model)
    print("------- with pretrained weights --------")
    model = load_pretrained_model(model, pretrained_path)
    infer(model)


if __name__ == "__main__":
    main("/home/jules/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth")