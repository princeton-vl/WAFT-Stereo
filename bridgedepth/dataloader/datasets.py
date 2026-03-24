import logging
import os
from pathlib import Path
from glob import glob
import os.path as osp
import numpy as np
import torch
import torch.utils
import torch.nn.functional as F
from typing import Iterable, Optional, TypeVar, List, Tuple, Union

from functools import partial
from ..utils import frame_utils, misc, dist_utils as comm
from .transforms import FlowAugmentor, SparseFlowAugmentor
from .base.easy_dataset import EasyDataset
from .sampler import InferenceSampler


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)


def verify_str_arg(
    value: T,
    arg: Optional[str] = None,
    valid_values: Optional[Iterable[T]] = None,
    custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, str):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)
    
    if valid_values is None:
        return value
    
    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)
    
    return value


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as fp:
        lines = [line.rstrip() for line in fp.readlines()]
    return lines


class StereoDataset(EasyDataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, resolution='F'):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None and "crop_size" in aug_params:
            if self.sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.disparity_list = []
        self.image_list = []
        if resolution not in ['F', 'H', 'Q']:
            raise ValueError("Non recognized resolution '{resolution}'".format(resolution=resolution))
        elif resolution == 'F':
            self.skip = 1
        elif resolution == 'H':
            self.skip = 2
        else:  # Q
            self.skip = 4

    def __getitem__(self, index):

        sample = {}
        if self.is_test:
            img1 = np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8)
            img2 = np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8)
            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
            
            sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()
            sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()
            sample['meta'] = self.image_list[index][0]
            return sample
        
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            initial_seed = torch.initial_seed() % 2**31
            if worker_info is not None:
                misc.seed_all_rng(initial_seed + worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp_path = self.disparity_list[index]
        if isinstance(disp_path, (tuple, list)):
            disp_path = disp_path[0]
        disp = self.disparity_reader(disp_path)
        if isinstance(disp, tuple):
            disp, valid = disp
            valid = valid & (disp > 0) & (disp < 1e3)
        else:
            valid = (disp > 0) & (disp < 1e3)
        
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()[:, ::self.skip, ::self.skip]
        sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()[:, ::self.skip, ::self.skip]
        sample['disp'] = torch.from_numpy(flow).permute(2, 0, 1).float()[:, ::self.skip, ::self.skip][0] / self.skip
        if self.sparse:
            valid = torch.from_numpy(valid)[::self.skip, ::self.skip]
        else:
            valid = (sample['disp'] > 0) & (sample['disp'] < 1e3)
            
        sample['valid'] = valid
        return sample
    
    def __len__(self):
        return len(self.image_list)
    
    def _scan_pairs(
        self,
        paths_left_pattern: str,
        paths_right_pattern: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        
        left_paths = list(sorted(glob(paths_left_pattern)))
        
        right_paths = List[Union[None, str]]
        if paths_right_pattern:
            right_paths = list(sorted(glob(paths_right_pattern)))
        else:
            right_paths = list(None for _ in left_paths)
        
        if not left_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_left_pattern}")
        
        if not right_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_right_pattern}")
        
        if len(left_paths) != len(right_paths):
            raise ValueError(
                f"Found {len(left_paths)} left files but {len(right_paths)} right files using:\n"
                f"left pattern: {paths_left_pattern}\n"
                f"right pattern: {paths_right_pattern}\n"
            )
        
        paths = list((left, right) for left, right in zip(left_paths, right_paths))
        return paths
    

class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/sceneflow', dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add Flythings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training', nonocc=False):
        super(ETH3D, self).__init__(aug_params, sparse=True, reader=partial(frame_utils.readDispETH3D, nonocc=nonocc))

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', split='training', image_set='2015', nonocc=False):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)
        occ_str = 'noc' if nonocc else 'occ'
        if image_set == '2012':
            root = osp.join(root, '2012')
            disp_prefix = osp.join(root, 'training', f'disp_{occ_str}')
            images1 = sorted(glob(osp.join(root, split, 'colored_0/*_10.png')))
            images2 = sorted(glob(osp.join(root, split, 'colored_1/*_10.png')))
        elif image_set == '2015':
            root = osp.join(root, '2015')
            disp_prefix = osp.join(root, 'training', f'disp_{occ_str}_0')
            images1 = sorted(glob(osp.join(root, split, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, split, 'image_3/*_10.png')))
        
        for img1, img2 in zip(images1, images2):
            self.image_list += [ [img1, img2] ]

        if split == 'testing':
            self.is_test = True
        else:
            self.disparity_list = sorted(glob(os.path.join(disp_prefix, '*_10.png')))
            
class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/middlebury', split='F', image_set='training', nonocc=False):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=partial(frame_utils.readDispMiddlebury, nonocc=nonocc))
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2005", "2006", "2014", "2021"]
        if split == "2005":
            scenes = [d for d in os.listdir(os.path.join(root, "2005")) if os.path.isdir(os.path.join(root, "2005", d))]
            for scene in scenes:
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        name = os.path.join(root, "2005", scene, scene.split('-')[0])
                        self.image_list += [[f"{name}/Illum{illum}/Exp{exp}/view1.png", f"{name}/Illum{illum}/Exp{exp}/view5.png"]]
                        self.disparity_list += [f"{name}/disp1.png"]
        elif split == "2006":
            scenes = [d for d in os.listdir(os.path.join(root, "2006")) if os.path.isdir(os.path.join(root, "2006", d))]
            for scene in scenes:
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        name = os.path.join(root, "2006", scene, scene.split('-')[0])
                        self.image_list += [[f"{name}/Illum{illum}/Exp{exp}/view1.png", f"{name}/Illum{illum}/Exp{exp}/view5.png"]]
                        self.disparity_list += [f"{name}/disp1.png"] 
        elif split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = [d for d in os.listdir(os.path.join(root, "2014")) if os.path.isdir(os.path.join(root, "2014", d))]
            for scene in scenes:
                if "imperfect" in scene:
                    continue
                for s in ["E","L",""]:
                    name = os.path.join(root, "2014", scene, scene)
                    self.image_list += [[f"{name}/im0.png", f"{name}/im1{s}.png"]]
                    self.disparity_list += [f"{name}/disp0.pfm"]
        elif split == "2021":
            dirs = [d for d in os.listdir(os.path.join(root, "2021")) if os.path.isdir(os.path.join(root, "2021", d))]
            for dir in dirs:
                if "ambient" not in dir:
                    continue
                name = os.path.join(root, "2021", dir, "data")
                scenes = [d for d in os.listdir(name) if os.path.isdir(os.path.join(name, d))]
                for scene in scenes:
                    for s in ["0", "1", "2", "3"]:
                        img0_path = os.path.join(name, scene, "ambient", "L0", f"im0e{s}.png")
                        img1_path = os.path.join(name, scene, "ambient", "L0", f"im1e{s}.png")
                        disp_path = os.path.join(root, "2021", "all", "data", scene, "disp0.pfm")
                        if os.path.exists(img0_path):
                            self.image_list += [[img0_path, img1_path]]
                            self.disparity_list += [disp_path]
        else:
            if image_set == 'training':
                lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            else:
                lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/testF/*"))))

            image1_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines]) if image_set == 'training' else [os.path.join(root, "MiddEval3", f'training{split}', 'Adirondack/disp0GT.pfm')]*len(image1_list)
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
                

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)
        
        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2
        
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
    
    
class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings', variant: str = "both"):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)
        root = Path(root)
        
        verify_str_arg(variant, "variant", valid_values=["single", "mixed", "both"])
        
        variants = {
            "single": ["single"],
            "mixed": ["mixed"],
            "both": ["single", "mixed"],
        }[variant]
        
        split_prefix = {
            "single": Path("*") / "*",
            "mixed": Path("*"),
        }
        
        for s in variants:
            left_img_pattern = str(root / s / split_prefix[s] / "*.left.jpg")
            right_img_pattern = str(root / s / split_prefix[s] / "*.right.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / split_prefix[s] / "*.left.depth.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)
               

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/TartanAir'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)
        root = Path(root)
        
        left_img_pattern = str(root / "*/*/*/image_left/*_left.png")
        right_img_pattern = str(root / "*/*/*/image_right/*_right.png")
        self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)
        
        left_disparity_pattern = str(root / "*/*/*/depth_left/*_left_depth.npy")
        self.disparity_list = self._scan_pairs(left_disparity_pattern, None)
        
        
class CREStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/CREStereo'):
        super().__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)
        root = Path(root)
        
        dirs = ["shapenet", "reflective", "tree", "hole"]
        
        for s in dirs:
            left_img_pattern = str(root / s / "*/*_left.jpg")
            right_img_pattern = str(root / s / "*/*_right.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*/*_left.disp.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)
            

class VirtualKitti2(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/VKITTI2'):
        super().__init__(aug_params, reader=frame_utils.readDispVKITTI)
        assert os.path.exists(root)
        root = Path(root)
        
        dirs = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        for s in dirs:
            left_img_pattern = str(root / s / "*" / "frames" / "rgb" / "Camera_0" / "rgb_*.jpg")
            right_img_pattern = str(root / s / "*" / "frames" / "rgb" / "Camera_1" / "rgb_*.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*" / "frames" / "depth" / "Camera_0" / "depth_*.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)


class CarlaHighres(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/HR-VS/carla-highres'):
        super().__init__(aug_params=aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, 'trainingF/*/im0.png')))
        image2_list = [im.replace('im0', 'im1') for im in image1_list]
        disp_list = [im.replace('im0.png', 'disp0GT.pfm') for im in image1_list]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/InStereo2K'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)
        root = Path(root)
        
        left_img_pattern = osp.join(root, "*/*" , "left.png")
        right_img_pattern = osp.join(root, "*/*" , "right.png")
        self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
        
        left_disparity_pattern = osp.join(root, "*/*" , "left_disp.png")
        self.disparity_list += self._scan_pairs(left_disparity_pattern, None)
            

class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/booster', split='train', resolution='F'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispBooster, resolution=resolution)
        assert os.path.exists(root)

        folder_list = sorted(glob(osp.join(root, split + '/balanced/*')))        
        for folder in folder_list:
            image1_list = sorted(glob(osp.join(folder, 'camera_00/im*.png')))
            image2_list = sorted(glob(osp.join(folder, 'camera_02/im*.png')))
            for img1, img2 in zip(image1_list, image2_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ osp.join(folder, 'disp_00.npy') ]

class FSD(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FSD', size=None):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispFSD)
        assert os.path.exists(root)
        root = Path(root)
        image1_list = sorted(glob(os.path.join(root, '*/*/*/left/rgb/*.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*/*/right/rgb/*.jpg')))
        self.disparity_list = sorted(glob(os.path.join(root, '*/*/*/left/disparity/*.png')))
        self.image_list = list(zip(image1_list, image2_list))
        if size is not None:
            self.image_list = self.image_list[:size]
            self.disparity_list = self.disparity_list[:size]

class Spring(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/spring', split='train_val'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSpring)
        assert os.path.exists(root)
        root = Path(root)
        root = os.path.join(root, split)
        image1_list = sorted(glob(os.path.join(root, '*/frame_left/*.png')))
        image2_list = sorted(glob(os.path.join(root, '*/frame_right/*.png')))
        self.disparity_list = sorted(glob(os.path.join(root, '*/disp1_left/*.dsp5')))
        self.image_list = list(zip(image1_list, image2_list))

class TartanGround(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/TartanGround'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanGround)
        assert os.path.exists(root)
        root = Path(root)
        scene = "*"
        # front, top, bottom, is okay
        self.image_list = []
        self.disparity_list = []
        normal = ['front', 'top', 'bottom']
        for cam in normal:
            left_img_pattern = str(root / scene / f"*/*/image_lcam_{cam}/*.png")
            right_img_pattern = str(root / scene / f"*/*/image_rcam_{cam}/*.png")
            self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)
            left_disparity_pattern = str(root / scene / f"*/*/depth_lcam_{cam}/*.png")
            self.disparity_list = self._scan_pairs(left_disparity_pattern, None)
        
        negative = ['back']
        for cam in negative:
            left_img_pattern = str(root / scene / f"*/*/image_rcam_{cam}/*.png")
            right_img_pattern = str(root / scene / f"*/*/image_lcam_{cam}/*.png")
            self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)
            left_disparity_pattern = str(root / scene / f"*/*/depth_rcam_{cam}/*.png")
            self.disparity_list = self._scan_pairs(left_disparity_pattern, None)

class UnrealStereo4K(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/UnrealStereo4K'):
        super().__init__(aug_params, reader=frame_utils.readDispUnrealStereo4K)
        assert os.path.exists(root)
        root = Path(root)
        left_img_pattern = str(root / "*/Image0/*.png")
        right_img_pattern = str(root / "*/Image1/*.png")
        self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)
        left_disparity_pattern = str(root / "*/Disp0/*.npy")
        self.disparity_list = self._scan_pairs(left_disparity_pattern, None)

class WMGStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/WMGStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispWMGStereo)
        assert os.path.exists(root)
        split = 'release_subset'
        root = Path(root)
        left_img_pattern = str(root / split / "*/*/*/Image/camera_0/*.png")
        right_img_pattern = str(root / split / "*/*/*/Image/camera_1/*.png")
        self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)
        left_disparity_pattern = str(root / split / "*/*/*/disparity/camera_0/*.npy")
        self.disparity_list = self._scan_pairs(left_disparity_pattern, None)

def build_train_loader(cfg):
    """ Create the data loader for the corresponding training set """
    crop_size = cfg.DATASETS.CROP_SIZE
    spatial_scale = cfg.DATASETS.SPATIAL_SCALE
    yjitter = cfg.DATASETS.YJITTER
    aug_params = {'crop_size': list(crop_size), 'min_scale': spatial_scale[0], 'max_scale': spatial_scale[1], 'do_flip': False, 'yjitter': yjitter}
    if cfg.DATASETS.SATURATION_RANGE is not None:
        aug_params["saturation_range"] = cfg.DATASETS.SATURATION_RANGE
    if cfg.DATASETS.IMG_GAMMA is not None:
        aug_params["gamma"] = cfg.DATASETS.IMG_GAMMA

    train_dataset = None
    logger = logging.getLogger(__name__)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.DATASETS.MUL)
    for dataset_name, mul in zip(cfg.DATASETS.TRAIN, cfg.DATASETS.MUL):
        if dataset_name.startswith('middlebury_'):
            split = dataset_name.split('_')[1]
            new_dataset = Middlebury(aug_params, split=split)
            logger.info(f"{len(new_dataset)} samples from {dataset_name}")
        elif dataset_name == 'sceneflow':
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass')
            new_dataset = final_dataset + clean_dataset
            logger.info(f"{len(new_dataset)} samples from SceneFlow")
        elif dataset_name.startswith('eth3d_'):
            nonocc = (dataset_name.split('_')[-1] == 'nonocc')
            new_dataset = ETH3D(aug_params, split='training', nonocc=nonocc)
            logger.info(f"{len(new_dataset)} samples from ETH3D")
        elif dataset_name.startswith('kitti_'):
            image_set = dataset_name.split('_')[1]
            split = dataset_name.split('_')[2]
            nonocc = (dataset_name.split('_')[-1] == 'nonocc')
            new_dataset = KITTI(aug_params, image_set=image_set, split=split, nonocc=nonocc)
            logger.info(f"{len(new_dataset)} samples from KITTI{image_set}_{split}")
        elif dataset_name == 'sintelstereo':
            new_dataset = SintelStereo(aug_params)
            logger.info(f"{len(new_dataset)} samples from SintelStereo")
        elif dataset_name == 'fallingthings':
            new_dataset = FallingThings(aug_params, variant='both')
            logger.info(f"{len(new_dataset)} samples from FallingThings")
        elif dataset_name == 'tartanair':
            new_dataset = TartanAir(aug_params)
            logger.info(f"{len(new_dataset)} samples from TartanAir")
        elif dataset_name == 'carlahighres':
            new_dataset = CarlaHighres(aug_params)
            logger.info(f"{len(new_dataset)} samples from Carla Highres")
        elif dataset_name == 'crestereo':
            new_dataset = CREStereo(aug_params)
            logger.info(f"{len(new_dataset)} samples from CREStereo")
        elif dataset_name == 'vkitti2':
            new_dataset = VirtualKitti2(aug_params)
            logger.info(f"{len(new_dataset)} samples from VirtualKitti2")
        elif dataset_name.startswith('booster'):
            resolution = dataset_name.split('_')[-1]
            new_dataset = Booster(aug_params, resolution=resolution)
            logger.info(f"{len(new_dataset)} samples from Booster")
        elif dataset_name == 'instereo2k':
            new_dataset = InStereo2K(aug_params)
            logger.info(f"{len(new_dataset)} samples from InStereo2K")
        elif dataset_name == 'fsd':
            new_dataset = FSD(aug_params)
            logger.info(f"{len(new_dataset)} samples from FSD")
        elif dataset_name == 'spring':
            new_dataset = Spring(aug_params)
            logger.info(f"{len(new_dataset)} samples from Spring")
        elif dataset_name == 'tartanground':
            new_dataset = TartanGround(aug_params)
            logger.info(f"{len(new_dataset)} samples from TartanGround")
        elif dataset_name == 'unrealstereo4k':
            new_dataset = UnrealStereo4K(aug_params)
            logger.info(f"{len(new_dataset)} samples from UnrealStereo4K")
        elif dataset_name == 'wmgstereo':
            new_dataset = WMGStereo(aug_params)
            logger.info(f"{len(new_dataset)} samples from WMGStereo")
        else:
            raise ValueError(f"Unrecognized dataset {dataset_name}")
        if mul > 0:
            new_dataset = mul * new_dataset
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    world_size = comm.get_world_size()
    total_batch_size = cfg.SOLVER.IMS_PER_BATCH
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({})".format(
        total_batch_size, world_size
    )
    batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank())
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)
    return train_loader, train_sampler


def build_val_loader(cfg, dataset_name):
    logger = logging.getLogger(__name__)
    if dataset_name == 'things':
        val_dataset = SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name.startswith('kitti_'):
        # perform validation using the KITTI (train) split
        image_set = dataset_name.split('_')[1]
        split = dataset_name.split('_')[2]
        nonocc = (dataset_name.split('_')[-1] == 'nonocc')
        val_dataset = KITTI(image_set=image_set, split=split, nonocc=nonocc)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name.startswith("eth3d"):
        nonocc = (dataset_name.split('_')[-1] == 'nonocc')
        val_dataset = ETH3D(split='training', nonocc=nonocc)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name.startswith("middlebury"):
        split = dataset_name.split('_')[1]
        nonocc = (dataset_name.split('_')[-1] == 'nonocc')
        val_dataset = Middlebury(split=split, nonocc=nonocc)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name == 'sceneflow':
        final_dataset = SceneFlowDatasets(dstype='frames_finalpass')
        clean_dataset = SceneFlowDatasets(dstype='frames_cleanpass')
        val_dataset = final_dataset + clean_dataset
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name == 'booster':
        val_dataset = Booster(resolution='Q')
        logger.info('Number of validation image pairs: %d' % len(val_dataset))

    world_size = comm.get_world_size()
    if world_size > 1:
        val_sampler = InferenceSampler(len(val_dataset))
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=True, drop_last=False,
                                             sampler=val_sampler)
    return val_loader