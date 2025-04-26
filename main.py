import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QColor, QPen,
                         QPainterPath, QFont)
from PyQt5.QtCore import Qt, QPoint, QRect, QPointF
import cv2
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn.functional as F

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- 工具函数 --------------------
def preprocess_image(image: Image.Image):
    """图像预处理管道"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).to(device)


def tensor_to_numpy(tensor: torch.Tensor):
    """将归一化的Tensor转换回numpy图像"""
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225])[:, None, None] \
             + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    tensor = tensor.permute(1, 2, 0).clamp(0, 1).numpy()
    return (tensor * 255).astype(np.uint8)


def gram_matrix(features: torch.Tensor):
    """计算Gram矩阵"""
    b, c, h, w = features.size()
    features = features.view(b * c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)


# -------------------- VGG模型定义 --------------------
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
                            'pool1', 'conv2_1', 'relu2_1', 'conv2_2',
                            'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                            'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
                            'relu4_2', 'conv4_3', 'relu4_3', 'pool4']

        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                name = f'conv{self._get_block(i)}_1'
            elif isinstance(layer, nn.ReLU):
                name = f'relu{self._get_block(i)}_1'
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{self._get_block(i)}'
            setattr(self, name, layer)

    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features

    def _get_block(self, i):
        if i < 4:
            return 1
        elif i < 9:
            return 2
        elif i < 16:
            return 3
        else:
            return 4


# -------------------- 交互式绘图组件 --------------------
class PaintArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)

        # 绘图参数
        self.drawing = False
        self.last_point = QPoint()
        self.current_path = QPainterPath()
        self.masks = {'hair': None, 'face': None, 'cloth': None}
        self.current_task = 'hair'
        self.colors = {
            'hair': (255, 0, 0, 80),
            'face': (0, 255, 0, 80),
            'cloth': (0, 0, 255, 80)
        }
        self.debug_points = []  # 用于坐标验证
        # 图像参数
        self.origin_img = None
        self.display_rect = QRect()
        self.display_scale = 1.0

    def set_image(self, np_img):
        """精确的图像显示和坐标计算"""
        try:
            # Print the shape of the image to debug
            print(f"Original Image Shape: {np_img.shape}")

            # If the image is grayscale (2D), convert it to RGB (3D)
            if len(np_img.shape) == 2:  # Grayscale image (2D)
                print("Converting grayscale image to RGB")
                np_img = np.stack([np_img] * 3, axis=-1)  # Convert to RGB by stacking the same data across 3 channels

            # Ensure that np_img is 3-dimensional
            if len(np_img.shape) != 3 or np_img.shape[2] != 3:
                raise ValueError(f"Expected an RGB image with 3 channels, but got shape {np_img.shape}")

            # Print the shape after conversion to verify
            print(f"Image Shape after conversion: {np_img.shape}")

            # Store the original image data
            self.origin_img = np_img.copy()

            # Unpack height, width, and channels, now safe to do because image is RGB
            h, w, _ = np_img.shape  # Unpack height, width, and channels

            # Ensure data is contiguous in memory
            np_img = np_img.copy()

            # Convert numpy array to bytes
            qimg = QImage(np_img.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Calculate display rectangle (centered)
            self.display_rect = QRect(
                (self.width() - pixmap.width()) // 2,  # Center horizontally
                (self.height() - pixmap.height()) // 2,  # Center vertically
                pixmap.width(),
                pixmap.height()
            )

            # Calculate accurate scaling factors
            self.display_scale_x = pixmap.width() / w
            self.display_scale_y = pixmap.height() / h

            # Create a canvas for drawing
            canvas = QPixmap(self.size())
            canvas.fill(Qt.transparent)

            painter = QPainter(canvas)
            # Draw the base image
            painter.drawPixmap(self.display_rect, pixmap)
            # Draw additional content like masks or paths
            self._draw_masks(painter)
            self._draw_path(painter)
            painter.end()

            self.setPixmap(canvas)
            self.debug_points.clear()  # Clear debug points
            self.orig_img_size = (np_img.shape[1], np_img.shape[0])  # (w, h)
            self.display_pixmap = QPixmap.fromImage(qimg).scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Calculate the actual display area (centered)
            self.display_rect = QRect(
                (self.width() - self.display_pixmap.width()) // 2,
                (self.height() - self.display_pixmap.height()) // 2,
                self.display_pixmap.width(),
                self.display_pixmap.height()
            )

            # Independently calculate XY scaling factors
            self.scale_x = self.display_pixmap.width() / self.orig_img_size[0]
            self.scale_y = self.display_pixmap.height() / self.orig_img_size[1]
        except Exception as e:
            # Debugging any exceptions
            print(f"Error occurred: {str(e)}")
            QMessageBox.critical(self.parent(), "图像错误", str(e))

    def mousePressEvent(self, event):
        """开始绘制"""
        if event.button() == Qt.LeftButton and self.origin_img is not None:
            self.drawing = True
            self.current_path = QPainterPath()
            pos = self._convert_pos(event.pos())
            self.current_path.moveTo(pos)
            self.last_point = pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            current_pos = self._convert_pos(event.pos())

            # 动态采样算法（基于移动速度）
            dx = current_pos.x() - self.last_point.x()
            dy = current_pos.y() - self.last_point.y()
            distance = math.hypot(dx, dy)

            if distance > 5:  # 像素移动超过5时增加采样
                steps = int(distance / 2) + 1
                x_step = dx / steps
                y_step = dy / steps

                for i in range(1, steps + 1):
                    interp_point = QPointF(
                        self.last_point.x() + x_step * i,
                        self.last_point.y() + y_step * i
                    )
                    self.current_path.lineTo(interp_point)

            self.last_point = current_pos
            self.update()

    def mouseReleaseEvent(self, event):
        """结束绘制"""
        if self.drawing:
            self.drawing = False
            try:
                if self.current_path.length() < 10:
                    raise ValueError("绘制区域过小")

                # 生成掩膜
                mask = self._create_mask()
                self._update_mask(mask)
                self.set_image(self.origin_img)

            except Exception as e:
                QMessageBox.critical(self.parent(), "错误", str(e))
            finally:
                self.current_path = QPainterPath()

    def _convert_pos(self, pos: QPoint) -> QPointF:
        """将界面坐标转换为图像坐标（亚像素级精度）"""
        # 转换为显示区域相对坐标
        relative_x = pos.x() - self.display_rect.x()
        relative_y = pos.y() - self.display_rect.y()

        # 转换为原始图像坐标（使用浮点运算）
        img_x = min(max(relative_x / self.scale_x, 0.0), self.orig_img_size[0] - 1e-6)
        img_y = min(max(relative_y / self.scale_y, 0.0), self.orig_img_size[1] - 1e-6)

        return QPointF(img_x, img_y)

    def _create_mask(self):
        h, w = self.origin_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 获取高精度路径点
        points = []
        for i in range(self.current_path.elementCount()):
            elem = self.current_path.elementAt(i)
            points.append([elem.x, elem.y])  # 已经是图像坐标系

        if len(points) < 3:
            return None

        # 数学闭合验证
        if np.linalg.norm(np.array(points[0]) - np.array(points[-1])) > 1e-6:
            points.append(points[0])

        # 转换为OpenCV格式（保留浮点精度）
        pts = np.array([points], dtype=np.float32)

        # 亚像素级填充（关键修正）
        cv2.fillPoly(
            mask,
            [np.round(pts).astype(np.int32)],  # 四舍五入到最近整数
            color=1,
            lineType=cv2.LINE_AA
        )

        # 精确后处理
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return (mask > 0.5).astype(np.uint8)

    def _update_mask(self, new_mask):
        """更新当前任务的掩膜"""
        if new_mask is None:
            return

        if self.masks[self.current_task] is None:
            self.masks[self.current_task] = new_mask
        else:
            self.masks[self.current_task] = np.clip(
                self.masks[self.current_task] + new_mask, 0, 1)

    def _draw_masks(self, painter):
        """绘制所有掩膜"""
        for task, mask in self.masks.items():
            if mask is not None:
                overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
                overlay[mask == 1] = self.colors[task]
                qmask = QImage(overlay.data, mask.shape[1], mask.shape[0],
                               QImage.Format_RGBA8888)
                painter.drawImage(self.display_rect, qmask)

    def _draw_path(self, painter):
        """精确路径绘制"""
        if not self.current_path.isEmpty():
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(255, 0, 0), 3, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)

            # 创建精确缩放路径
            scaled_path = QPainterPath()
            for i in range(self.current_path.elementCount()):
                elem = self.current_path.elementAt(i)
                x = self.display_rect.x() + elem.x * self.display_scale_x
                y = self.display_rect.y() + elem.y * self.display_scale_y

                if i == 0:
                    scaled_path.moveTo(x, y)
                else:
                    scaled_path.lineTo(x, y)

            # 绘制调试点
            for px, py, ix, iy in self.debug_points:
                painter.drawEllipse(QPoint(px, py), 3, 3)

            painter.drawPath(scaled_path)

    def paintEvent(self, event):
        super().paintEvent(event)

        # 绘制调试信息
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 1))

        # 显示坐标转换信息
        text = f"Scale: ({self.scale_x:.3f}, {self.scale_y:.3f})\n"
        text += f"Display Rect: {self.display_rect.getCoords()}"
        painter.drawText(10, 20, text)

        # 绘制基准标记
        if self.origin_img is not None:
            # 图像中心点
            center_x = self.display_rect.x() + self.display_pixmap.width() / 2
            center_y = self.display_rect.y() + self.display_pixmap.height() / 2
            painter.drawEllipse(QPoint(int(center_x), int(center_y)), 5, 5)



    def clear_current_mask(self):
        """清空当前任务掩膜"""
        self.masks[self.current_task] = None
        self.set_image(self.origin_img)


# -------------------- 主应用程序 --------------------
class StyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手动区域风格迁移")
        self.setGeometry(100, 100, 1000, 700)

        # 初始化参数
        self.tasks = ['hair', 'face', 'cloth']
        self.current_task_idx = 0
        self.content_tensor = None
        self.style_tensor = None

        # 初始化UI
        self.init_ui()

        # 加载默认图像
        self.load_images()

    def init_ui(self):
        """初始化用户界面"""
        # 绘图区域
        self.paint_area = PaintArea(self)

        # 控制按钮
        self.prev_btn = QPushButton("上一个区域 (←)", self)
        self.next_btn = QPushButton("下一个区域 (→)", self)
        self.clear_btn = QPushButton("清空当前区域 (C)", self)
        self.transfer_btn = QPushButton("开始风格迁移", self)

        # 状态显示
        self.status_label = QLabel(self)
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))

        # 布局设置
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.transfer_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.paint_area)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

        # 信号连接
        self.prev_btn.clicked.connect(self.prev_task)
        self.next_btn.clicked.connect(self.next_task)
        self.clear_btn.clicked.connect(self.clear_current_task)
        self.transfer_btn.clicked.connect(self.start_transfer)

        self.update_status()

    def load_images(self):
        """加载内容图和风格图"""
        try:
            # 内容图
            content_img = Image.open("content/img_1.png").convert("RGB")
            self.content_tensor = preprocess_image(content_img)
            content_np = tensor_to_numpy(self.content_tensor)
            self.paint_area.set_image(content_np)

            # 风格图
            style_img = Image.open("style/img_1.png").convert("RGB")
            self.style_tensor = preprocess_image(style_img)

        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"图像加载失败: {str(e)}")
            sys.exit(1)

    def update_status(self):
        """更新状态显示"""
        text = f"当前区域: {self.tasks[self.current_task_idx].upper()} | " \
               f"进度: {self.current_task_idx + 1}/{len(self.tasks)}"
        self.status_label.setText(text)
        self.paint_area.current_task = self.tasks[self.current_task_idx]

    def prev_task(self):
        """切换到上一个区域"""
        if self.current_task_idx > 0:
            self.current_task_idx -= 1
            self.update_status()
            self.paint_area.set_image(tensor_to_numpy(self.content_tensor))

    def next_task(self):
        """切换到下一个区域"""
        if self.current_task_idx < len(self.tasks) - 1:
            self.current_task_idx += 1
            self.update_status()
            self.paint_area.set_image(tensor_to_numpy(self.content_tensor))

    def clear_current_task(self):
        """清空当前区域"""
        self.paint_area.clear_current_mask()
        QMessageBox.information(self, "提示",
                                f"已清空{self.tasks[self.current_task_idx]}区域")

    def start_transfer(self):
        """执行风格迁移"""
        try:
            # 验证所有掩膜
            for task in self.tasks:
                if self.paint_area.masks[task] is None:
                    raise ValueError(f"请先完成{task}区域的标注！")
                if np.sum(self.paint_area.masks[task]) == 0:
                    raise ValueError(f"{task}区域标注为空！")

            # 准备掩膜
            masks = {
                task: torch.from_numpy(mask).float().to(device)
                for task, mask in self.paint_area.masks.items()
            }
            masks['background'] = 1 - sum(masks.values())

            # 初始化模型
            vgg = models.vgg19(pretrained=True).features.to(device).eval()

            # 风格迁移优化
            input_img = self.content_tensor.clone().requires_grad_(True)
            optimizer = optim.LBFGS([input_img], lr=0.5)

            style_weight = 1e6
            content_weight = 1
            max_iter = 300
            iter_count = 0

            def closure():
                nonlocal iter_count
                optimizer.zero_grad()
                total_loss = 0

                # 提取特征
                content_features = vgg(self.content_tensor)
                style_features = vgg(self.style_tensor)
                input_features = vgg(input_img)

                # 内容损失
                content_loss = F.mse_loss(input_features[2], content_features[2])

                # 风格损失
                style_loss = 0
                for i in [0, 1, 2, 3, 4]:
                    style_loss += F.mse_loss(gram_matrix(input_features[i]),
                                             gram_matrix(style_features[i]))

                # 区域加权
                for region, mask in masks.items():
                    region_content = self.content_tensor * mask
                    region_style = self.style_tensor * mask
                    region_input = input_img * mask

                    # 区域内容损失
                    content_loss += 0.3 * F.mse_loss(vgg(region_input)[2],
                                                     vgg(region_content)[2])

                    # 区域风格损失
                    for i in [0, 1, 2, 3, 4]:
                        style_loss += 0.3 * F.mse_loss(
                            gram_matrix(vgg(region_input)[i]),
                            gram_matrix(vgg(region_style)[i]))

                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss.backward()

                if iter_count % 50 == 0:
                    print(f"Iteration {iter_count}: Loss={total_loss.item():.2f}")
                iter_count += 1
                return total_loss

            # 优化循环
            while iter_count < max_iter:
                optimizer.step(closure)

            # 显示结果
            result = tensor_to_numpy(input_img)
            plt.figure(figsize=(12, 6))
            plt.subplot(121), plt.imshow(tensor_to_numpy(self.content_tensor))
            plt.title("Content Image")
            plt.subplot(122), plt.imshow(result)
            plt.title("Styled Image")
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "迁移错误", str(e))


if __name__ == "__main__":
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("使用CPU运行")

    # 启动应用
    app = QApplication(sys.argv)
    window = StyleTransferApp()
    window.show()
    sys.exit(app.exec_())