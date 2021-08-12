# -*- coding: utf-8 -*-

### This block is same as HW3 ###
# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

"""## **Dataset, Data Loader, and Transforms** *(similar to HW3)*

Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.

Here, since our data are stored in folders by class labels, we can directly apply **torchvision.datasets.DatasetFolder** for wrapping data without much effort.

Please refer to [PyTorch official website](https://pytorch.org/vision/stable/transforms.html) for details about different transforms.

---
**The only diffference with HW3 is that the transform functions are different.**
"""

### This block is similar to HW3 ###
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.

train_tfm = transforms.Compose([
  # Resize the image into a fixed shape (height = width = 142)
	# transforms.Resize((142, 142)),
  # transforms.RandomHorizontalFlip(),
  # transforms.RandomRotation(15),
  # transforms.RandomCrop(128),
	# transforms.ToTensor(),

  # 參考HW3 TA solution
    transforms.Resize((142, 142)),
    # transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)), 
    transforms.RandomPerspective(distortion_scale=0.2,p=0.5), 
    transforms.RandomCrop(128),
    transforms.ToTensor(),

])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 142)
    transforms.Resize((142, 142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

### This block is similar to HW3 ###
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
# batch_size = 64
batch_size = 32

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



def conv_bn(inp,oup,kernel_size=3,stride=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
        nn.MaxPool2d(2, 2, 0),     
    )

# Merge Depthwise and Pointwise Convolution (without )
def dwpw_conv(in_chs, out_chs, kernel_size=3, stride=1, padding=1,pool=True):
    if pool:
        return nn.Sequential(
            nn.Conv2d(in_chs, in_chs, kernel_size, stride, padding, groups=in_chs),
            nn.BatchNorm2d(in_chs),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2, 2, 0),     
    
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_chs),
            nn.ReLU6(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_chs, in_chs, kernel_size, stride, padding, groups=in_chs),
            nn.BatchNorm2d(in_chs),
            nn.ReLU6(inplace=True),    
            
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_chs),
            nn.ReLU6(inplace=True),
        )

class StudentNet(nn.Module):
    def __init__(self):
      super(StudentNet, self).__init__()

      # ---------- TODO ----------
      # Modify your model architecture

      self.cnn = nn.Sequential(
        # ---------- TA ----------
        # nn.Conv2d(3, 32, 3), 
        # nn.BatchNorm2d(32),
        # nn.ReLU(),
        # nn.Conv2d(32, 32, 3),  
        # nn.BatchNorm2d(32),
        # nn.ReLU(),
        # nn.MaxPool2d(2, 2, 0),     

        # nn.Conv2d(32, 64, 3), 
        # nn.BatchNorm2d(64),
        # nn.ReLU(),
        # nn.MaxPool2d(2, 2, 0),     

        # nn.Conv2d(64, 100, 3), 
        # nn.BatchNorm2d(100),
        # nn.ReLU(),
        # nn.MaxPool2d(2, 2, 0),
        # ---------- TA ----------
        conv_bn(3, 32),
        dwpw_conv(32, 64),
        dwpw_conv(64, 64,pool=False),
        dwpw_conv(64, 128),
        dwpw_conv(128, 128,pool=False),
        dwpw_conv(128, 128),
        dwpw_conv(128, 256,pool=False),
        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(256, 11),
      )
      
    def forward(self, x):
      out = self.cnn(x)
      out = out.view(out.size()[0], -1)
      return self.fc(out)

"""## **Model Analysis**

Use `torchsummary` to get your model architecture (screenshot or pasting text are allowed.) and numbers of 
parameters, these two information should be submit to your NTU Cool questions.

Note that the number of parameters **should not greater than 100,000**, or you'll get penalty in this homework.

"""
if __name__ == '__main__':       
    
    from torchsummary import summary
    
    student_net = StudentNet()
    summary(student_net, (3, 128, 128), device="cpu")
    
    """## **Knowledge Distillation**
    
    <img src="https://i.imgur.com/H2aF7Rv.png=100x" width="500px">
    
    Since we have a learned big model, let it teach the other small model. In implementation, let the training target be the prediction of big model instead of the ground truth.
    
    ## **Why it works?**
    * If the data is not clean, then the prediction of big model could ignore the noise of the data with wrong labeled.
    * The labels might have some relations. Number 8 is more similar to 6, 9, 0 than 1, 7, for example.
    
    
    ## **How to implement?**
    * $Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{Original Loss})$
    * Note that the logits here should have passed softmax.
    """
    
    # ref: https://zhuanlan.zhihu.com/p/79437280 參考之乎連結時做 soft loss in knowledge distillation
    # KD_loss = nn.KLDivLoss()(F.softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1)) * alpha
    
    def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5):
        T = 20
        hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 
        # ---------- TODO ----------
        # Complete soft loss in knowledge distillation
        soft_loss = 0 
        soft_loss = (alpha * T * T)* nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),F.softmax(teacher_outputs/T, dim=1))
        return hard_loss + soft_loss
    
    """## **Teacher Model Setting**
    We provide a well-trained teacher model to help you knowledge distillation to student model.
    Note that if you want to change the transform function, you should consider  if suitable for this well-trained teacher model.
    * If you cannot successfully gdown, you can change a link. (Backup link is provided at the bottom of this colab tutorial).
    
    """
    
    # Download teacherNet
    # !gdown --id '1zH1x39Y8a0XyOORG7TWzAnFf_YPY8e-m' --output teacher_net.ckpt
    # Load teacherNet
    teacher_net = torch.load('./teacher_net.ckpt')
    teacher_net.eval()
    
    """## **Generate Pseudo Labels in Unlabeled Data**
    
    Since we have a well-trained model, we can use this model to predict pseudo-labels and help the student network train well. Note that you 
    **CANNOT** use well-trained model to pseudo-label the test data. 
    
    
    ---
    
    **AGAIN, DO NOT USE TEST DATA FOR PURPOSE OTHER THAN INFERENCING**
    
    * Because If you use teacher network to predict pseudo-labels of the test data, you can only use student network to overfit these pseudo-labels without train/unlabeled data. In this way, your kaggle accuracy will be as high as the teacher network, but the fact is that you just overfit the test data and your true testing accuracy is very low. 
    * These contradict the purpose of these assignment (network compression); therefore, you should not misuse the test data.
    * If you have any concerns, you can email us.
    
    """
    
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize a model, and put it on the device specified.
    student_net = student_net.to(device)
    teacher_net = teacher_net.to(device)
    
    # Whether to do pseudo label.
    do_semi = True
    
    def get_pseudo_labels(dataset, model):
        loader = DataLoader(dataset, batch_size=batch_size*3, shuffle=False, pin_memory=True)
        pseudo_labels = []
        for batch in tqdm(loader):
            # A batch consists of image data and corresponding labels.
            img, _ = batch
    
            # Forward the data
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(img.to(device))
                pseudo_labels.append(logits.argmax(dim=-1).detach().cpu())
            # Obtain the probability distributions by applying softmax on logits.
        pseudo_labels = torch.cat(pseudo_labels)
        # Update the labels by replacing with pseudo labels.
        for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
            dataset.samples[idx] = (img, pseudo_label.item())
        return dataset
    
    if do_semi:
        # Generate new trainloader with unlabeled set.
        unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net)
        concat_dataset = ConcatDataset([train_set, unlabeled_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    
    torch.cuda.empty_cache()
    import gc
    del unlabeled_set,concat_dataset, train_set, valid_set, test_set
    gc.collect()
    
    """## **Training** *(similar to HW3)*
    
    You can finish supervised learning by simply running the provided code without any modification.
    
    The function "get_pseudo_labels" is used for semi-supervised learning.
    It is expected to get better performance if you use unlabeled data for semi-supervised learning.
    However, you have to implement the function on your own and need to adjust several hyperparameters manually.
    
    For more details about semi-supervised learning, please refer to [Prof. Lee's slides](https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/semi%20(v3).pdf).
    
    Again, please notice that utilizing external data (or pre-trained model) for training is **prohibited**.
    
    ---
    **The only diffference with HW3 is that you should use loss in  knowledge distillation.**
    
    
    
    """
    
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.AdamW(student_net.parameters(), lr=0.001)

    # The number of training epochs.
    n_epochs = 300
    # student_net.load_state_dict(torch.load('student_net.bin'))
    
    # for epoch in range(n_epochs):
    #     if epoch == 150:
    #         optimizer = torch.optim.AdamW(student_net.parameters(), lr=1e-4)
    #     if epoch == 230:
    #         optimizer = torch.optim.SGD(student_net.parameters(), lr=0.0003, momentum=0.9)
    #     # ---------- Training ----------
    #     # Make sure the model is in train mode before training.
    #     student_net.train()
    
    #     # These are used to record information in training.
    #     train_loss = []
    #     train_accs = []
    
    #     # Iterate the training set by batches.
    #     for batch in tqdm(train_loader):
      
    #         # A batch consists of image data and corresponding labels.
    #         imgs, labels = batch
    
    #         # Forward the data. (Make sure data and model are on the same device.)
    #         logits = student_net(imgs.to(device))
    #         # Teacher net will not be updated. And we use torch.no_grad
    #         # to tell torch do not retain the intermediate values
    #         # (which are for backpropgation) and save the memory.
    #         with torch.no_grad():
    #           soft_labels = teacher_net(imgs.to(device))
            
    #         # Calculate the loss in knowledge distillation method.
    #         loss = loss_fn_kd(logits, labels.to(device), soft_labels)
    
    #         loss.backward()
    #         # Compute the gradients for parameters.

            
    #         # Clip the gradient norms for stable training.
    #         grad_norm = nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=10)
    
    #         # Update the parameters with computed gradients.
    #         optimizer.step()
    #         # Gradients stored in the parameters in the previous step should be cleared out first.
    #         optimizer.zero_grad()
    #         # Compute the accuracy for current batch.
    #         acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    
    #         # Record the loss and accuracy.
    #         train_loss.append(loss.item())
    #         train_accs.append(acc)
    
    #     # The average loss and accuracy of the training set is the average of the recorded values.
    #     train_loss = sum(train_loss) / len(train_loss)
    #     train_acc = sum(train_accs) / len(train_accs)
    #     # pbar.set_description(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    #     # Print the information.
    #     # print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    
    
    #     # ---------- Validation ----------
    #     # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    #     student_net.eval()
    
    #     # These are used to record information in validation.
    #     valid_loss = []
    #     valid_accs = []
    
    #     # Iterate the validation set by batches.
    #     # pbar = 
    #     for batch in tqdm(valid_loader):
            
    #         # A batch consists of image data and corresponding labels.
    #         imgs, labels = batch
    
    #         # We don't need gradient in validation.
    #         # Using torch.no_grad() accelerates the forward process.
    #         with torch.no_grad():
    #           logits = student_net(imgs.to(device))
    #           soft_labels = teacher_net(imgs.to(device))
    #         # We can still compute the loss (but not the gradient).
    #         loss = loss_fn_kd(logits, labels.to(device), soft_labels)
    
    #         # Compute the accuracy for current batch.
    #         acc = (logits.argmax(dim=-1) == labels.to(device)).float().detach().cpu().view(-1).numpy()
    
    #         # Record the loss and accuracy.
    #         valid_loss.append(loss.item())
    #         valid_accs += list(acc)
    
    #     # The average loss and accuracy for entire validation set is the average of the recorded values.
    #     valid_loss = sum(valid_loss) / len(valid_loss)
    #     valid_acc = sum(valid_accs) / len(valid_accs)
    
    #     # Print the information.
    #     # pbar.set_description(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    #     print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f} [ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    #     # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    #     if epoch % 20 ==0 and not epoch ==0:
    #       torch.save(student_net.state_dict(), f'student_net_{epoch}.bin')
    #     torch.save(student_net.state_dict(), f'student_net.bin')
    # torch.save(student_net.state_dict(), f'student_net_finish.bin')
    """## **Testing** *(same as HW3)*
    
    For inference, we need to make sure the model is in eval mode, and the order of the dataset should not be shuffled ("shuffle=False" in test_loader).
    
    Last but not least, don't forget to save the predictions into a single CSV file.
    The format of CSV file should follow the rules mentioned in the slides.
    
    ### **WARNING -- Keep in Mind**
    
    Cheating includes but not limited to:
    1.   using testing labels,
    2.   submitting results to previous Kaggle competitions,
    3.   sharing predictions with others,
    4.   copying codes from any creatures on Earth,
    5.   asking other people to do it for you.
    
    Any violations bring you punishments from getting a discount on the final grade to failing the course.
    
    It is your responsibility to check whether your code violates the rules.
    When citing codes from the Internet, you should know what these codes exactly do.
    You will **NOT** be tolerated if you break the rule and claim you don't know what these codes do.
    
    """
    
    ### This block is same as HW3 ###
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    import glob
    predictions_lst = []
    for model in glob.glob("student_net/*.bin"):
        student_net.load_state_dict(torch.load(model))
        print(model)
        student_net.eval()
        
        # Initialize a list to store the predictions.
        predictions = []
        # Iterate the testing set by batches.
        for batch in tqdm(test_loader):
            # A batch consists of image data and corresponding labels.
            # But here the variable "labels" is useless since we do not have the ground-truth.
            # If printing out the labels, you will find that it is always 0.
            # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
            # so we have to create fake labels to make it work normally.
            imgs, labels = batch
        
            # We don't need gradient in testing, and we don't even have labels to compute loss.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = student_net(imgs.to(device))
        
            # Take the class with greatest logit as prediction and record it.
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        predictions_lst.append(predictions)
        ### This block is same as HW3 ###
        # Save predictions into the file.
    from scipy import stats
    m = stats.mode(np.array(predictions_lst))[0]
    with open("predict.csv", "w") as f:
    
        # The first row must be "Id, Category"
        f.write("Id,Category\n")
    
        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(m[0]):
             f.write(f"{i},{pred}\n")
