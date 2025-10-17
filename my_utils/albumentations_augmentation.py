"""
Albumentations-based Advanced Augmentation for Semantic Segmentation
Albumentations 기반 고급 증강 기법들
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


def get_advanced_augmentation(config):
    """고급 증강 파이프라인"""
    
    # 기본 증강 (PC1과 동일)
    basic = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.Rotate(limit=5, p=0.5)
    ]
    
    advanced = []
    
    # 1. 날씨 시뮬레이션 (현실적 환경)
    if config.get('use_weather', False):
        weather_p = config.get('weather_p', 0.3)
        advanced.append(
            A.OneOf([
                A.RandomRain(
                    slant_range=(-10, 10),
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=5,
                    brightness_coefficient=0.7,
                    p=1
                ),
                A.RandomFog(
                    fog_coef_range=(0.1, 0.3),
                    alpha_coef=0.1,
                    p=1
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    p=1
                ),
            ], p=weather_p)
        )
    
    # 2. 노이즈 추가 (센서 노이즈 시뮬레이션)
    if config.get('use_noise', False):
        noise_p = config.get('noise_p', 0.3)
        advanced.append(
            A.OneOf([
                A.GaussNoise(p=1),
                A.ISONoise(p=1),
                A.MultiplicativeNoise(p=1),
            ], p=noise_p)
        )
    
    # 3. Blur 효과 (모션, 초점 등)
    if config.get('use_blur', False):
        blur_p = config.get('blur_p', 0.3)
        advanced.append(
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1),
                A.MedianBlur(blur_limit=5, p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
            ], p=blur_p)
        )
    
    # 4. CutOut/GridDropout (일부 영역 가리기)
    if config.get('use_cutout', False):
        cutout_p = config.get('cutout_p', 0.3)
        advanced.append(
            A.OneOf([
                A.CoarseDropout(
                    p=1
                ),
                A.GridDropout(ratio=0.2, p=1),
            ], p=cutout_p)
        )
    
    # 5. 기하학적 변형
    if config.get('use_geometric', False):
        geometric_p = config.get('geometric_p', 0.3)
        advanced.append(
            A.OneOf([
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    p=1
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    p=1
                ),
                A.OpticalDistortion(
                    distort_limit=0.1,
                    p=1
                ),
            ], p=geometric_p)
        )
    
    # 6. 색상 변형
    if config.get('use_color', False):
        color_p = config.get('color_p', 0.3)
        advanced.append(
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1
                ),
                A.ChannelShuffle(p=1),
            ], p=color_p)
        )
    
    return A.Compose([
        A.Resize(height=1024, width=1024),  # 크기 고정
    ] + basic + advanced + [
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                   std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_validation_transform():
    """Validation용 Transform (증강 없음)"""
    return A.Compose([
        A.Resize(height=1024, width=1024),  # 크기 고정
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                   std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class AlbumentationsSegmentationTransform:
    """Albumentations 기반 Segmentation Transform"""
    
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.75, 1.25], 
                 # 기본 증강
                 horizontal_flip_p=0.5, brightness_limit=0.2, contrast_limit=0.2, rotation_limit=5,
                 # 고급 증강 설정
                 use_weather=False, weather_p=0.3,
                 use_noise=False, noise_p=0.3,
                 use_blur=False, blur_p=0.3,
                 use_cutout=False, cutout_p=0.3,
                 use_geometric=False, geometric_p=0.3,
                 use_color=False, color_p=0.3):
        
        self.crop_size = crop_size
        self.scale_range = scale_range
        
        # 기본 증강 파라미터
        self.horizontal_flip_p = horizontal_flip_p
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.rotation_limit = rotation_limit
        
        # 고급 증강 파라미터
        self.use_weather = use_weather
        self.weather_p = weather_p
        self.use_noise = use_noise
        self.noise_p = noise_p
        self.use_blur = use_blur
        self.blur_p = blur_p
        self.use_cutout = use_cutout
        self.cutout_p = cutout_p
        self.use_geometric = use_geometric
        self.geometric_p = geometric_p
        self.use_color = use_color
        self.color_p = color_p
        
        # Albumentations Transform 생성
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Albumentations Transform 생성"""
        config = {
            'use_weather': self.use_weather,
            'weather_p': self.weather_p,
            'use_noise': self.use_noise,
            'noise_p': self.noise_p,
            'use_blur': self.use_blur,
            'blur_p': self.blur_p,
            'use_cutout': self.use_cutout,
            'cutout_p': self.cutout_p,
            'use_geometric': self.use_geometric,
            'geometric_p': self.geometric_p,
            'use_color': self.use_color,
            'color_p': self.color_p,
        }
        
        return get_advanced_augmentation(config)
    
    def __call__(self, image, label):
        """Transform 적용"""
        # PIL Image를 numpy로 변환
        import numpy as np
        img_np = np.array(image)
        label_np = np.array(label)
        
        # Albumentations 적용
        transformed = self.transform(image=img_np, mask=label_np)
        
        return transformed['image'], transformed['mask']


class AlbumentationsValidationTransform:
    """Validation용 Transform (증강 없음)"""
    
    def __init__(self, crop_size=[1024, 1024]):
        self.crop_size = crop_size
        self.transform = get_validation_transform()
    
    def __call__(self, image, label):
        """Transform 적용"""
        # PIL Image를 numpy로 변환
        import numpy as np
        img_np = np.array(image)
        label_np = np.array(label)
        
        # Albumentations 적용
        transformed = self.transform(image=img_np, mask=label_np)
        
        return transformed['image'], transformed['mask']
