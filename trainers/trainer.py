import torch
from torch import optim
from models.lsmf_net import LSMFNet  # 你可以整合上面的模块
from losses import ContentLoss, StyleLoss, TotalLoss
from datasets.lsmf_dataset import LSMFDataset
from torch.utils.data import DataLoader


def train():
    # 数据加载
    train_dataset = LSMFDataset("data/train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 模型、损失函数、优化器
    model = LSMFNet()
    content_loss_fn = ContentLoss()
    style_loss_fn = StyleLoss()
    total_loss_fn = TotalLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(100):  # 示例100个epoch
        for batch_idx, (content, style) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传递
            output = model(content, style)
            content_loss = content_loss_fn(output, content)
            style_loss = style_loss_fn(output, style)
            total_loss = total_loss_fn(output, content, style)

            # 反向传播
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/100], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {total_loss.item()}")


if __name__ == "__main__":
    train()
