### ImageNet 이미지의 평균 및 표준편차로 Normalize 된 Pytorch Tensor 타입의 이미지를 Unnormalize하기 위한 클래스
### 이미지 입력으로 활용된 Pytorch Tensor 타입의 이미지를 활용하는 것이 아니라면 활용 불필요

### Unnormalize Image Tensor Values for Visualization
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor