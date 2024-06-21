import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import ColorJitter

# 这个类主要是为增强数据的泛化性
class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, no_eraser_aug=True):
        # 空间增强参数
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # 翻转增广参数
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # 光度增强参数
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)

        self.asymmetric_color_aug_prob = 0.2

        if no_eraser_aug:
            # 我们禁用了橡皮擦, 因为在我们的实验中没有观察到明显的改善
            self.eraser_aug_prob = -1
        else:
            self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ 光度增大 """

        # 不对称的;不对等的
        '''
            具体来说,这个参数的作用是决定在color_transform方法中是否应用不对称的颜色增强。
            在color_transform方法中,根据asymmetric_color_aug_prob的值,
            程序会以self.asymmetric_color_aug_prob的概率来决定是否应用不对称的颜色增强操作.
            如果随机数小于self.asymmetric_color_aug_prob,则会对img1和img2进行不对称的颜色增强操作;
            否则,将对这两个图像进行对称的颜色增强操作。
        '''
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # 对称的
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    """
        if bounds is None:：如果bounds参数为None, 则将bounds设置为默认值[50, 100]. bounds是用来指定遮挡区域大小的参数, 表示遮挡区域的高度和宽度范围.
        ht, wd = img1.shape[:2]：获取img1图像的高度和宽度。
        if np.random.rand() < self.eraser_aug_prob:以self.eraser_aug_prob的概率判断是否应用遮挡操作。如果随机数小于self.eraser_aug_prob，则执行以下操作。
        mean_color = np.mean(img2.reshape(-1, 3), axis=0)：计算img2图像的均值颜色，将图像展平为一个二维数组，然后计算每个通道（RGB）的均值，得到mean_color。
        for _ in range(np.random.randint(1, 3)):：随机选择1到2次遮挡操作。
        在每次遮挡操作中：
        随机生成遮挡区域的起始点(x0, y0)和遮挡区域的高度和宽度(dx, dy)。
        将img2图像中的遮挡区域（img2[y0:y0 + dy, x0:x0 + dx, :]）的像素值设置为mean_color，即用均值颜色填充遮挡区域。
        最后，返回经过可能的遮挡操作后的img1和img2图像。
    """
    def eraser_transform(self, img1, img2, bounds=None):

        if bounds is None:
            bounds = [50, 100]
        ht, wd = img1.shape[:2]
        # 以self.eraser_aug_prob的概率判断是否应用遮挡操作. 如果随机数小于self.eraser_aug_prob，则执行以下操作.
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    '''
        这个方法主要用于对图像进行随机的空间变换操作，包括缩放、拉伸、翻转和裁剪，以增加数据的多样性和丰富性，有助于提高模型的泛化能力和鲁棒性
    '''
    def spatial_transform(self, img1, dpt1, img2, dpt2, flow, occlusion=None):
        # 随机抽样放缩
        ht, wd = img1.shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # print("dpt0.shape"+str(dpt1.shape))

        if np.random.rand() < self.spatial_aug_prob:    # 经过这里的depth均变为shape均变为3维
            # 重新缩放图像
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dpt1 = cv2.resize(dpt1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dpt1 = np.expand_dims(dpt1, axis=2)     # 维度为1会降维
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dpt2 = cv2.resize(dpt2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dpt2 = np.expand_dims(dpt2, axis=2)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

            if occlusion is not None:
                occlusion = cv2.resize(occlusion, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            
            # print("dpt1.shape"+str(dpt1.shape))    
        else:
            dpt1 = np.expand_dims(dpt1, axis=2)     # 维度为1会降维
            dpt2 = np.expand_dims(dpt2, axis=2)
            # print("dpt2.shape"+str(dpt1.shape))

        # print("dpt3.shape"+str(dpt1.shape))

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # 水平翻转
                img1 = img1[:, ::-1]
                dpt1 = dpt1[:, ::-1]
                img2 = img2[:, ::-1]
                dpt2 = dpt2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

                if occlusion is not None:
                    occlusion = occlusion[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # 垂直翻转
                img1 = img1[::-1, :]
                dpt1 = dpt1[::-1, :]
                img2 = img2[::-1, :]
                dpt2 = dpt2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

                if occlusion is not None:
                    occlusion = occlusion[::-1, :]

        # 如果没有裁剪
        if img1.shape[0] - self.crop_size[0] > 0:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        else:
            y0 = 0
        if img1.shape[1] - self.crop_size[1] > 0:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        else:
            x0 = 0

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        dpt1 = dpt1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        dpt2 = dpt2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        if occlusion is not None:
            occlusion = occlusion[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            return img1, dpt1, img2, dpt2, flow, occlusion

        return img1, dpt1, img2, dpt2, flow

    def __call__(self, img1, dpt1, img2, dpt2, flow, occlusion=None):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)

        if occlusion is not None:
            img1, dpt1, img2, dpt2, flow, occlusion = self.spatial_transform(
                img1, dpt1, img2, dpt2, flow, occlusion)
        else:
            img1, dpt1, img2, dpt2, flow = self.spatial_transform(img1, dpt1, img2, dpt2, flow)

        img1 = np.ascontiguousarray(img1)
        dpt1 = np.ascontiguousarray(dpt1)
        img2 = np.ascontiguousarray(img2)
        dpt2 = np.ascontiguousarray(dpt2)
        flow = np.ascontiguousarray(flow)

        if occlusion is not None:
            occlusion = np.ascontiguousarray(occlusion)
            return img1, dpt1, img2, dpt2, flow, occlusion

        return img1, dpt1, img2, dpt2, flow


# 数据集中稀疏光流的数据增强
class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, no_eraser_aug=True):
        # 空间增强参数
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # 翻转拓展数据
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # 光度增强参数,函数可以随机调整图像亮度，增强训练泛化性
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2

        if no_eraser_aug:
            # 我们禁用了 eraser aug, 因为在我们的实验中没有观察到明显的改善
            self.eraser_aug_prob = -1
        else:
            self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, dpt1, img2, dpt2, flow, valid):
        # 随机抽样尺度

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dpt1 = cv2.resize(dpt1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dpt2 = cv2.resize(dpt2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                dpt1 = dpt1[:, ::-1]
                img2 = img2[:, ::-1]
                dpt2 = dpt2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        # 随机剪裁
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        dpt1 = dpt1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        dpt2 = dpt2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, dpt1, img2, dpt2, flow, valid

    def __call__(self, img1, dpt1, img2, dpt2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)

        img1, dpt1, img2, dpt2, flow, valid = self.spatial_transform(img1, dpt1, img2, dpt2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        dpt1 = np.ascontiguousarray(dpt1)
        img2 = np.ascontiguousarray(img2)
        dpt2 = np.ascontiguousarray(dpt2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        return img1, dpt1, img2, dpt2, flow, valid
