import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from Utils import FrameUtils
from data.Transforms import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None,
                 sparse=False,
                 load_occlusion=False,
                 ):
        self.augmentor = None
        self.sparse = sparse

        # 这里主要是对数据集进行相对应的拓展: 翻转,变换图像的亮度,图像拉伸等
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []     # 光流list
        self.image_list = []    # 图像list
        self.depth_list = []    # 深度list
        self.extra_info = []

        self.load_occlusion = load_occlusion
        self.occ_list = []

    # 训练和测试数据集
    def __getitem__(self, index):
        # print('Pair = ', self.depth_list[index][0], self.depth_list[index][1])
        # 是否是测试数据:测试数据则设置随机种子保证数据加载的随机性
        if self.is_test:
            img1 = FrameUtils.read_gen(self.image_list[index][0])
            img2 = FrameUtils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            return img1, img2, self.extra_info[index]
        # 如果不是测试数据
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        # 计算实际索引值
        index = index % len(self.image_list)
        valid = None
        # 如果设置了self.sparse: 读取光流数据
        if self.sparse:
            # 读取KITTI数据集
            flow, valid = FrameUtils.readFlowKITTI(self.flow_list[index])  # [H, W, 2], [H, W]
        else:
            flow = FrameUtils.read_gen(self.flow_list[index])
        # 是否加载遮挡
        if self.load_occlusion:
            occlusion = FrameUtils.read_gen(self.occ_list[index])  # [H, W], 0 or 255 (occluded)

        img1 = FrameUtils.read_gen(self.image_list[index][0])
        img2 = FrameUtils.read_gen(self.image_list[index][1])
        # 如果传入的是.dpt文件则读取深度，如果传入的是pfm文件则用标志位判别是视差图
        dpt1 = FrameUtils.read_gen(self.depth_list[index][0], False)
        dpt2 = FrameUtils.read_gen(self.depth_list[index][1], False)

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.load_occlusion:
            occlusion = np.array(occlusion).astype(np.float32)

        '''
            这段代码的作用是确保img1和img2具有相同的通道数，
            以便在后续处理中保持数据的一致性。
            如果img1是灰度图像，代码会将其扩展为具有三个通道的图像。
            如果img1已经是一个具有三个或更多通道的图像，代码会保留前三个通道
        '''
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # 数据集增强, 颜色改变处理不影响Depth，但是翻转、放缩、剪裁需要重设Depth
        if self.augmentor is not None:
            if self.sparse:
                img1, dpt1, img2, dpt2, flow, valid = self.augmentor(img1, dpt1, img2, dpt2, flow, valid)
            else:
                if self.load_occlusion:
                    img1, dpt1, img2, dpt2, flow, occlusion = self.augmentor(img1, dpt1, img2, dpt2, flow, occlusion=occlusion)
                else:
                    img1, dpt1, img2, dpt2, flow = self.augmentor(img1, dpt1, img2, dpt2, flow)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        dpt1 = torch.from_numpy(dpt1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        dpt2 = torch.from_numpy(dpt2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.load_occlusion:
            occlusion = torch.from_numpy(occlusion)  # [H, W]

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        # mask out occluded pixels
        if self.load_occlusion:
            # non-occlusion: 0, occlusion: 255
            noc_valid = 1 - occlusion / 255.  # 0 or 1

            return img1, dpt1, img2, dpt2, flow, valid.float(), noc_valid.float()

        return img1, dpt1, img2, dpt2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root='F:\datasets\Sintel',
                 dstype='clean',
                 load_occlusion=False,
                 ):
        super(MpiSintel, self).__init__(aug_params,
                                        load_occlusion=load_occlusion,
                                        )

        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')

        if load_occlusion:
            occlusion_root = osp.join(root, split, 'occlusions')

        if split == 'test':
            self.is_test = True

        # print('depth_root = ' , depth_root);

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # 场景和帧ID
                self.depth_list += [[depth_list[i], depth_list[i + 1]]]
            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                if load_occlusion:
                    self.occ_list += sorted(glob(osp.join(occlusion_root, scene, '*.png')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None,
                 split='train',
                 root='datasets/FlyingChairs_release/data',
                 ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chairs_split.txt')
        split_list = np.loadtxt(split_file, dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None,
                 root='datasets/FlyingThings3D',
                 dstype='frames_cleanpass',
                 test_set=False,
                 validate_subset=True,
                 ):
        super(FlyingThings3D, self).__init__(aug_params)

        img_dir = root
        flow_dir = root

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                if test_set:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TEST/*/*')))
                else:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                if test_set:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TEST/*/*')))
                else:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])


                # 代码中的循环遍历了两个目录列表 image_dirs 和 flow_dirs，使用 zip() 函数将它们进行配对。每个配对包含一个图像目录路径和一个光流目录路径
                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]



        # validate on 1024 subset of test set for fast speed
        if test_set and validate_subset:
            num_val_samples = 1024
            all_test_samples = len(self.image_list)  # 7866

            stride = all_test_samples // num_val_samples
            remove = all_test_samples % num_val_samples

            # uniformly sample a subset
            self.image_list = self.image_list[:-remove][::stride]
            self.flow_list = self.flow_list[:-remove][::stride]

class Monkaa(FlowDataset):
    def __init__(self, aug_params=None,
                 root='F:\datasets\Monkaa',
                 dstype='clean',
                 ):
        super(Monkaa, self).__init__(aug_params)

        image_root = osp.join(root, dstype, 'frames_cleanpass')
        flow_root = osp.join(root, 'flow', 'optical_flow')
        disparity_root = osp.join(root, 'dispartity', 'disparity')
        # print('Scene = ', image_scene)

        # for f in os.listdir(image_root):
        #     pathlist = osp.join(f, 'left')
        # print('PathList = ', pathlist)

        for direction in ['into_future', 'into_past']:      # 光流方向
            image_dirs = sorted([osp.join(image_root, f, 'left') for f in os.listdir(image_root)])
            disparity_dirs = sorted([osp.join(disparity_root, f, 'left') for f in os.listdir(disparity_root)])
            flow_dirs = sorted([osp.join(flow_root, f, direction, 'left') for f in os.listdir(flow_root)])
            # 代码中的循环遍历了两个目录列表 image_dirs 和 flow_dirs，使用 zip() 函数将它们进行配对。每个配对包含一个图像目录路径和一个光流目录路径
            for idir, ddir, fdir in zip(image_dirs, disparity_dirs, flow_dirs):
                images = sorted(glob(osp.join(idir, '*.png')))          # RGB图
                disparities = sorted(glob(osp.join(ddir, '*.pfm')))     # 视差图
                flows = sorted(glob(osp.join(fdir, '*.pfm')))           # 光流图
                for i in range(len(flows) - 1):
                    if direction == 'into_future':
                        self.image_list += [[images[i], images[i + 1]]]
                        self.depth_list += [disparity_dirs[i], disparity_dirs[i + 1]]
                        self.flow_list += [flows[i]]
                    elif direction == 'into_past':
                        self.image_list += [[images[i + 1], images[i]]]
                        self.depth_list += [disparity_dirs[i + 1], disparity_dirs[i]]
                        self.flow_list += [flows[i + 1]]

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root='datasets/KITTI',
                 ):
        super(KITTI, self).__init__(aug_params, sparse=True,
                                    )
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


def build_train_dataset(args):
    """ Create the data loader for the corresponding training set """
    # if args.stage == 'chairs':
    #     aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
    #
    #     train_dataset = FlyingChairs(aug_params, split='training')

    if args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}

        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        # 1041 pairs for clean and final each
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')  # 40302

        monkaa = Monkaa(aug_params, dstype='clean')

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')


        # aug_params = {'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}
        #
        # kitti = KITTI(aug_params=aug_params)  # 200

        # aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}
        #
        # hd1k = HD1K(aug_params=aug_params)  # 1047

        # train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

        train_dataset = sintel_clean + sintel_final   # 将每个数据对象重复添加到train_dataset 100次 用于控制当前数据集的权重

    # elif args.stage == 'kitti':
    #     aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
    #
    #     train_dataset = KITTI(aug_params, split='training',
    #                           )
    else:
        raise ValueError(f'stage {args.stage} is not supported')

    return train_dataset
