---
title: 计算机视觉：无人驾驶汽车的前沿
separator: <!--s-->
verticalSeparator: <!--v-->
theme: simple
highlightTheme: monokai-sublime
css: 
    - custom.css
    - dark.css
revealOptions:
    transition: 'slide'
    transitionSpeed: fast
    center: false
    slideNumber: "c/t"
    width: 1000
---

<div class="middle center">
<div style="width: 100%">

# 计算机视觉：无人驾驶汽车的前沿

<hr/>

By [胥涵坤](https://xiu-zju.me)

</div>
</div>

Note:
大家好，我是胥涵坤，今天我给大家带来的展示是关于计算机视觉在无人驾驶汽车控制论当中的应用。
<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part.1 传感系统

视觉感知、环境理解和决策支持

</div>
</div>
Note:
首先，让我们来看一看传感系统在整个无人驾驶汽车的控制中发挥的作用。
<!--v-->

## 无人驾驶汽车的控制系统

在无人驾驶汽车的整个控制系统中，**传感系统**起到了十分重要的作用，因为它负责实时感知车辆周围的环境，包括道路状况、行人、其他车辆及障碍物等，为决策系统提供准确的数据支持，从而保障行车安全和驾驶效率。

<center>
<img src="./1.png" alt="示例图片" width="800">
</center>

Note:
在无人驾驶汽车的整个控制系统中，**传感系统**起到了不可或缺的作用。从下面的图可以看出，传感系统负责实时接收车辆周围的环境信息，包括道路状况、行人、其他车辆及障碍物等，然后为决策系统提供准确的数据支持，从而保障行车安全和驾驶效率。

<!--v-->


## 传感系统的作用

- 识别障碍物和行人
- 车道识别
- 交通标志识别

以下是车道检测和交通标志识别的示例图：
<center>
<img src="./2.png" alt="示例图片" width="380">
<img src="./3.png" alt="示例图片" width="440">
</center>

Note:
传感系统具体要做哪些事呢？首先，最基本的，车辆要准确地识别出障碍物以及行人，避免发生碰撞。车辆还要知道自己该沿什么方向行驶，这就是车道识别。另外，无人驾驶车辆也要遵守交通规则，所以识别交通标志也是必要的。传感系统要实现包括但不限于以上的三个功能。

<!--v-->
## 传感系统的实现

计算机视觉！

<center>
<img src="./download.jpg" alt="示例图片" width="380">
</center>

计算机视觉在自动驾驶汽车中主要用于视觉感知和环境理解。视觉感知包括对车辆、行人、道路标志、交通信号等目标的检测和识别，以及对车辆的速度、距离等属性的估计。环境理解包括对道路环境的分析和理解，以及对驾驶行为的决策支持。

Note:
如何实现传感系统呢？当今最热门的技术便是用人工智能赋能的计算机视觉技术。利用计算机视觉，我们可以实现自动驾驶汽车视觉感知、环境理解等操作。

<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part.2 计算机视觉
利用计算机视觉来实现传感系统
</div>
</div>

Note:
下面就让我来粗略地介绍一下计算机视觉。

<!--v-->

## 概览

[计算机视觉](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)是一门研究如何使机器“看”的科学，更进一步的说，就是指用摄影机和计算机代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图像处理，用计算机处理成为更适合人眼观察或传送给仪器检测的图像。

那么，计算机视觉具体该怎么运用于无人驾驶汽车的传感工作呢？主要工作分为以下2个部分：

1. 图像处理
2. 目标检测与对象识别

Note:
计算机视觉是一门研究如何使机器“看”的科学，更进一步的说，就是指用摄影机和计算机代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图像处理的技术。要想实现无人驾驶汽车的传感功能，具体有两个步骤：一是处理摄像头传来的图像，二是对处理后的图像进行目标检测和对象识别。

<!--v-->

## 使用工具
语言选择：Python

<center>
<img src="./python.png" alt="示例图片" width="150">
</center>

库选择：OpenCV

<center>
<img src="./opencv.png" alt="示例图片" width="150">
</center>

Note:
要想自己动手实现类似的功能，我们可以利用python来实现。工具我们选择OpenCV。这是一个跨平台的计算机视觉库，可用于开发实时的图像处理、计算机视觉以及模式识别程序。

<!--s-->

<div class="middle center">
<div style="width: 100%">

# 第一步：图像处理
包括灰度转换、滤波、边缘检测等
</div>
</div>

Note:
刚才我们讲到，传感系统的工作主要分为两部分，一个是图像处理，还有一个目标检测和对象识别。我们先来谈谈图像处理。摄像头刚拍出来的图片是不好直接拿给机器去使用的。我们要对其进行一系列的处理。处理过程包括灰度转换、滤波和边缘检测等等。

<!--v-->
## 灰度转换
灰度变换是指根据某种目标条件按一定变换关系逐点改变源图像中每个像素灰度值的方法。目的是为了改善画质，使图像显示效果更加清晰。

彩色图像转为灰度图像的常用方法是通过加权平均法，也称为Y'CbCr颜色空间转换。这是因为人眼对绿色敏感度最高，对红色次之，对蓝色最不敏感。对此，我们有如下公式：

$$Gray(x,y)=0.299R(x,y)+0.587G(x,y)+0.114B(x,y)$$

<center>
<img src="./lao8.png" alt="示例图片" width="300">
<img src="./gray.jpg" alt="示例图片" width="300">
</center>

Note:
灰度变换是指根据某种目标条件按一定变换关系逐点改变源图像中每个像素灰度值的方法。为什么要进行灰度转换呢？这是为了改善画质，使图像显示效果更加清晰。彩色图像转为灰度图像的常用方法是通过加权平均法。人眼对绿色敏感度最高，对红色次之，对蓝色最不敏感。对此，我们有如下公式。这是处理后的效果。

<!--v-->
## 灰度转换实现代码

```python
import cv2
from PIL import Image

#读取彩色图像
color_img = cv2.imread(r'./lao8.png')

#在窗口中显示图像，该窗口和图像的原始大小自适应
cv2.imshow('original image',color_img)

#cvtColor的第一个参数是处理的图像，第二个是RGB2GRAY
gray_img=cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)

#gray_img此时还是二维矩阵表示,所以要实现array到image的转换
gray=Image.fromarray(gray_img)

gray.save('gray.jpg')
cv2.imshow('Gray Image',gray_img)

cv2.waitKey(0)

```
Note:
这是灰度转换的代码。由于时间原因我就不在这里展示了。思路很简单，就是调用一下库，套一下公式。

<!--v-->
## 滤波

由于成像系统、传输介质和记录设备等的不完善，数字图像在其形成、传输记录过程中往往会受到多种噪声的污染。[滤波](https://baike.baidu.com/item/%E6%BB%A4%E6%B3%A2/2938301)是用于减少图像噪声的技术，常用的滤波方法包括平均滤波、中值滤波、高斯滤波等。

滤波操作的要求:

1. 不能损坏图像轮廓及边缘
2. 图像视觉效果应当更好

<center>
<img src="./4.png" alt="示例图片" width="570">
</center>

Note:
由于成像系统、传输介质和记录设备等的不完善，数字图像在其形成、传输记录过程中往往会受到多种噪声的污染。因此我们要进行滤波。常用的滤波方法包括平均滤波、中值滤波、高斯滤波等。

<!--v-->
## 平均滤波
平均值滤波算法是比较常用，也比较简单的滤波算法。在滤波时，将N个周期的采样值计算平均值。当N取值较大时，滤波后的信号比较平滑，但是灵敏度差；相反N取值较小时，滤波平滑效果差，但灵敏度好。

- 优点：算法简单，对周期性干扰有良好的抑制作用，平滑度高，适用于高频振动的系统。
- 缺点：对异常信号的抑制作用差，无法消除脉冲干扰的影响。

$$f_{avg}(x, y) = \frac{1}{9} \sum_{i=-1}^1 \sum_{j=-1}^1 f(x+i, y+i)$$

Note:
平均滤波，顾名思义，就是取平均。对于一个3x3的窗口，正中间的像素值就是这个3x3的窗口的像素值的平均。公式如下：优点是算法简单，对周期性干扰有良好的抑制作用，平滑度高，适用于高频振动的系统。缺点是对异常信号的抑制作用差，无法消除脉冲干扰的影响。

<!--v-->
## 中值滤波
中值滤波是将图像窗口内像素值按值排序后选择中间值作为目标像素值的方法。对于一个3x3的窗口，中值滤波可以通过以下步骤实现：

- 对窗口内像素值排序，得到排序后的像素值序列
- 选择序列中的中间值作为目标像素值：$f_{median}(x,y)$

<center>
<img src="./5.png" alt="示例图片" width="570">
</center>

Note:
中值滤波是将图像窗口内像素值按值排序后选择中间值作为目标像素值的方法。对于一个3x3的窗口，他的方法是将这九个值进行排序，取中值。优点是消除杂散噪声点而不会或较小程度地造成边缘模糊。缺点是对于图像中含有较多点、线、尖角细节的，不适宜采用中值滤波。

<!--v-->
## 边缘检测
边缘检测是用于识别图像中锐边斜角的技术，常用的边缘检测算法包括罗尔边缘检测、艾伯尔边缘检测、卡尔曼滤波器等。

此处仅展示用法，原理不过多细究。
<center>
<img src="./7.png" alt="示例图片" width="470">
<img src="./heibai.png" alt="示例图片" width="470">
</center>

Note:
接下来就是边缘检测。边缘检测是用于识别图像中锐边斜角的技术，常用的边缘检测算法包括罗尔边缘检测、艾伯尔边缘检测、卡尔曼滤波器等。通过边缘检测，我们可以得到如下效果：

<!--v-->
## 边缘检测方法

- 罗尔边缘检测：

基于图像的梯度和拉普拉斯操作符的差分来检测边缘的方法。

$$
L(x, y) = |\nabla f(x, y)| =$$ 
$$\sqrt{(f(x+1, y) - f(x-1, y))^2 + (f(x, y+1) - f(x, y-1))^2}
$$
- 艾伯尔边缘检测：

是基于图像的梯度和拉普拉斯操作符的差分来检测边缘的方法，与罗尔边缘检测的区别在于使用了平滑操作。
$$
A(x, y) = |\nabla f(x, y)| − k \times \Delta f(x,y)$$ 

~~现实中直接用就行了，没必要弄清原理（逃~~

Note:
这里是边缘检测的两种方法，一个是罗尔边缘检测，另一个是艾伯尔边缘检测。大家看一看就行了。我感觉如果你不是研究这个方向的话，只需要学会黑盒使用就可以了，没必要弄清原理。

<!--v-->
## 代码实现
```python
import cv2
import numpy as np

# 读取并转换为灰度图像
image = cv2.imread('./7.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用中值滤波
median_filtered = cv2.medianBlur(gray, 5)

# 应用Canny边缘检测
canny_edges = cv2.Canny(gray, 50, 150)

# 绘制边缘检测结果
cv2.imshow('Canny Edges', canny_edges)

# 等待用户按任意键继续
cv2.waitKey(0)
```

Note:
这里是实现的代码。可以看到，写代码的时候只需要直接调用就可以了。想这里，灰度转换、利用中值滤波、应用边缘检测，获得图像。

<!--s-->

<div class="middle center">
<div style="width: 100%">

# 第二步：目标检测
包括R-CNN、Fast R-CNN、Faster R-CNN等
</div>
</div>

Note:
那么通过图像处理，我们已经获得了可以喂给计算机的图像。接下来我们要做的便是对其进行目标检测。

<!--v-->
## 意义
第一步的图像检测中，我们已经让机器得到了一个相对易识别的图像，接下来我们要让机器知道给它的图像是什么。目标检测就是在一幅图片中找到目标物体，给出目标的类别和位置，如下图所示：

<center>
<img src="./9.jpg" alt="示例图片" width="730">
</center>

Note:
目标检测就是在一幅图片中找到目标物体，给出目标的类别和位置，如下图所示：

<!--v-->
## 检测方法

目标检测方法分为**One-Stage**（一步检测算法）和**Two-Stage**（两步检测算法）两种。

- 两步检测算法是把整个检测过程分为两个步骤，第一步提取一些可能包含目标的候选框，第二步再从这些候选框中找出具体的目标并微调候选框。
- 一步检测算法则是省略了这个过程，直接在原始图片中预测每个目标的类别和位置。

两步检测最经典的就是Faster R-CNN三部曲。

1. R-CNN
2. Fast R-CNN
3. Faster R-CNN

<!--v-->
## R-CNN

R-CNN是Region-based Convolutional Neural Networks的缩写，是一种基于区域的卷积神经网络检测方法。R-CNN的主要步骤包括：

- 使用Selective Search算法生成候选的目标区域。
- 将生成的候选区域作为输入，使用卷积神经网络进行特征提取。
- 使用支持向量机（SVM）分类器对提取出的特征进行分类和回归，得到目标的类别和边界框坐标。

<center>
<img src="./10.png" alt="示例图片" width="630">
</center>

Note:
首先我们来看R-CNN。包括...其中这个Selective Search是用于生成候选区域的算法，通过图像分割将图像划分为初始区域，然后基于颜色、纹理、大小等特征层次合并区域，生成约2000个可能包含目标的候选框。由于接下来要对这2000歌区域都使用卷积神经网络，所以很慢。这个巨大的计算量令R-CNN难以在实际应用中被广泛采用。

<!--v-->
## Fast R-CNN
Fast R-CNN是一种改进的R-CNN方法，通过将候选区域生成和特征提取过程合并，提高检测速度。Fast R-CNN的主要步骤包括：

- 使用卷积神经网络对输入图像进行特征提取。
- 使用卷积神经网络的卷积层和池化层的输出作为候选区域的生成。
- 使用支持向量机（SVM）分类器对提取出的特征进行分类和回归，得到目标的类别和边界框坐标。

<center>
<img src="./11.png" alt="示例图片" width="630">
</center>

Note:
R-CNN的主要性能瓶颈在于需要对每个提议区域独立抽取特征。由于这些区域通常有大量重叠，独立的特征抽取会导致大量的重复计算。Fast R-CNN对R-CNN的一个主要改进在于只对整个图像做卷积神经网络的前向计算。

<!--v-->
## Faster R-CNN
Faster R-CNN是一种进一步改进的R-CNN方法，通过引入Region Proposal Network（RPN）来自动生成候选区域，进一步提高检测速度。Faster R-CNN的主要步骤包括：

- 使用卷积神经网络对输入图像进行特征提取。
- 使用Region Proposal Network（RPN）自动生成候选区域。
- 使用卷积神经网络对生成的候选区域进行特征提取。
- 使用支持向量机（SVM）分类器对提取出的特征进行分类和回归，得到目标的类别和边界框坐标。

<center>
<img src="./12.png" alt="示例图片" width="370">
</center>

Note:
Faster R-CNN 在 Fast R-CNN 的基础上引入了区域提议网络（RPN），用深度学习替代了 Selective Search，实现了候选区域的高效生成，使整个检测过程端到端训练，更快、更精确，进一步提升了效率和性能。三部曲大概就是这样的。有一个点就是三种方法的共同的步骤：...这是什么意思呢？支持向量机（SVM）是一种监督学习模型，用于分类和回归，通过寻找最佳的决策边界（超平面）将数据分开。在目标检测中，SVM 利用提取的特征对候选区域进行分类（判断是否包含目标及其类别），同时通过回归微调边界框的位置和大小，以更准确地定位目标。

<!--v-->
## 实现效果
将上述得到的结果进行分割，我们可以得到以下结果：

<center>
<img src="./13.jpg" alt="示例图片" width="700">
</center>


<!--s-->

<div class="middle center">
<div style="width: 100%">

# 实现效果
十分粗略地试了一下......
</div>
</div>

<!--v-->
## 简单程序
我试着使用opencv和预训练的模型实现了一个简单的人脸检测程序。无人驾驶车辆的检测模型应该是同理的，但是会复杂许多。

<center>
<img src="./14.jpg" alt="示例图片" width="600">
</center>

Note:
为什么我只是用矩形将人脸框住而不是用边缘呢？原因很简单，因为我不太会。

<!--v-->
## 代码实现
```python
import cv2
import os

# 加载预训练的人脸级联分类器
cascade_path = os.path.join('.../haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_path):
    print(f"Error: {cascade_path} not found.")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
```
<!--v-->
```python
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用级联分类器检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 为每个检测到的人脸绘制一个矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 显示结果
    cv2.imshow('Faces found', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头, 关闭所有窗口
cap.release()
cv2.destroyAllWindows()
```

<!--s-->

<div class="middle center">
<div style="width: 100%">

# 谢谢大家！

<hr/>

By [胥涵坤](https://xiu-zju.me)

<span style="background: linear-gradient(to right, orange, yellow, green); -webkit-background-clip: text; color: transparent; font-size: 34px; font-weight: bold;">
Any Questions?
</span>

</div>
</div>