import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, device):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有训练和验证两个阶段
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播+优化(只在训练阶段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model



def evaluate_model(model, dataloaders, dataset_sizes, device):
    model.eval()  # 设置为评估模式
    running_corrects = 0

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / dataset_sizes['test']
    print(f'Validation Accuracy: {acc:.4f}')


def main():
    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')

    # 检查是否可以使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=True)

    # 修改最后的全连接层以匹配你的分类任务
    num_classes = 3  # 3个类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 将模型移动到GPU（如果可用）
    model = model.to(device)

    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    data_dir = './'
    image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x]) for x in
                      ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in
                   ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    criterion = nn.CrossEntropyLoss()

    # 只微调最后的全连接层参数，其余层参数保持冻结
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 训练模型
    model = train_model(model, criterion, optimizer, num_epochs=15, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)

    # 保存模型
    torch.save(model.state_dict(), './model/model.pth')
    print('model saved!')

    # 观察模型在测试集上的效果
    evaluate_model(model, dataloaders, dataset_sizes, device)

if __name__ == '__main__':
    main()