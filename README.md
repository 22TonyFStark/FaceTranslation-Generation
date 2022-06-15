# FaceTranslation-Generation


#### **CVPR 2022 Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation.**

[[PDF](https://arxiv.org/abs/2203.12707)][[Github](https://github.com/batmanlab/MSPC)]

*[Yanwu Xu](http://xuyanwu.github.io/), Shaoan Xie, Wenhao Wu, Kun Zhang, [Mingming Gong](https://mingming-gong.github.io/), [Kayhan Batmanghelich](https://kayhan.dbmi.pitt.edu/).*

​      这篇论文的目的是为了缩小源域与目标域在空间形变上的差异，比如如下图所示的真人与动漫，二者除了颜色（style）的差异外，在形状（content）上有很大的不同。因此额外增加了一个T网络来辅助网络去拟合源于到目标域的形状变换的分布。

<img src="C:\Users\79072\AppData\Roaming\Typora\typora-user-images\image-20220508111341479.png" alt="image-20220508111341479" style="zoom: 50%;" />

​      整体结构如上图所示：首先网络包括四个子网络：
$$
G:风格迁移网络，输入源域图片，输出风格迁移后目标域图片  \\
D:监督风格迁移效果的辨别器 \\
T:空间变换网络，输入图片（源域or目标域），输出空间变换后的图片 \\
D_T:监督空间变换效果的辨别器
$$
​       并且包括了三个对抗训练：
$$
\begin{aligned}
&1.\ \min _{G} \max _{D} \mathbb{E}_{y \sim P_{Y}} \log D(y)+\mathbb{E}_{x \sim P_{X}} \log (1-D(G(x))) \\
&2.\ \min _{G, T} \max_{D_{T}} \mathbb{E}_{y \sim P_{Y}} \log D_{T}(T(y))+\mathbb{E}_{x \sim P_{X}} \log \left(1-D_{T}(G(T(x)))\right), \\
&3.\ \min _{G} \max _{T} \mathbb{E}_{x \sim P_{X}}\|T(G(x)), G(T(x))\|_{1} \\
&\text { s.t. } \frac{1}{a}<\frac{\left|p_{i} p_{j}\right|}{\left|q_{i} q_{j}\right|}<a, i \neq j \&-b<\sum_{i=1}^{n} p_{i}<b .
\end{aligned}
$$
​        其中1是original的风格迁移GAN训练过程，G生成图片来欺骗D，目的是拟合style变化分布；

​        2是一个新的空间变换GAN的训练过程，G和T一起生成图片来欺骗DT，目的是拟合形状变化分布；

​        3是新的G和T的对抗训练过程，这个过程的目标函数是T(G(X))和G(T(x))的L1距离，即先风格迁移再空间变换的结果A与先空间变换再风格迁移的结果B两者的L1距离，G要使A与B的距离最小化而T要使这个距离最大化。这里的理解是T可以视为对图片一个数据增强，目的是让G对形变更加鲁棒、并且拟合源域A与目标域B的空间形变上的差异。

> *T* plays an important role to generate maximum perturbation that tries to confuse *G* and enable *G* to be more robust across different I2I tasks. Furthermore, the deforming property of *T* can help align the spatial distribution in an unsupervised manner between the source images X and the target images *Y* by scaling, rotating, cropping noisy background, etc. 

​          优化过程中有一个约束条件，是对T网络的形变程度进行一个约束，以防止出现不合理的形变。

​		 以上就是此论文的overview，主要创新是增加了一个形变网络，其他的是original的GAN流程。下面具体介绍一下形变网络T。

​		 形变网络T是使用了Spatial Transformation Network(STN)，这个网络是定义了可导的形变函数，并且使用神经网络来学习形变函数的超参数。

![image-20220508115512105](C:\Users\79072\AppData\Roaming\Typora\typora-user-images\image-20220508115512105.png)

​		  

​		首先SFT网络可以是一个CNN，其输入为原图片（shape 3x256x256），输出为图片的形变（shape 256x256，每个通道的形变保持一致），形变和图片尺寸一样，其每一个像素是输入图片对应像素在形变之后的位置。

​        而形变图片这样构造：对于形变后图片上每一个位置i的像素点，其颜色值是输入图片所有像素点颜色值的一个线性组合，这个线性组合的系数由形变和插值方法确定。

​		下式或许更清晰一些：M是输入的图片，p是输入图片的一个位置，Φ是一个神经网络（输入p得到得到形变后的位置p'），理想情况是直接让形变后位置p'的颜色值等于形变前p的颜色值。而由于p‘是网络的输出是float值，而图片是像素化的，所以需要进行插值。比如下面的线性插值，用float p规整化的四个邻居像素点的颜色值作为p'的颜色值（类似于mask RCNN的RoI Align）：
$$
M(\phi(p))=\sum_{q \in \mathcal{Z}(\phi(p))} M(q) \prod_{d \in\{x, y, z\}}\left(1-\left|\phi_{d}(p)-q_{d}\right|\right)
$$
​        具体可以参考STN的提出论文https://proceedings.neurips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf

  		

#### **CVPR 2022 A Style-aware Discriminator for Controllable Image Translation.**

[[PDF](https://arxiv.org/abs/2203.15375)][[Github](https://github.com/kunheek/style-aware-discriminator)]
*Kunhee Kim, Sanghun Park, Eunyeong Jeon, Taehun Kim, Daijin Kim.*

​		这篇论文的目的是显示地让判别器更注重style。特点是把content和style解耦了，并使用了原型学习：

​		1.把生成器看成两个部分G_e和G_d，一个encoder，一个decoder，把判别器D也看成是一个encoder。

​		2.G_e是一个encoder，更具体地是对content信息的encoder，而判别器是对style信息的encoder，这两个模型输入是图片，输出一个相应信息的embedding。

​		3.G_d是一个decoder，其输入是2中的style embedding和content embedding，输出是风格迁移后的图片。

​		4.判别器除了负责对style info进行encode之外，还负责传统的判别。

![image-20220508133208070](C:\Users\79072\AppData\Roaming\Typora\typora-user-images\image-20220508133208070.png)

​		(a)把D当作一个style encoder，输入一张目标域图片，输出是目标域的style embedding。

​		(b)训练辨别器：1个对抗损失和1个自监督Swap predict损失。

​		由于辨别器需要作为一个style encoder，需要训练其对style的表征能力。所以使用了对比学习SwAV里的训练方法，我的理解是通过对一张图片进行content层面上的变换（比如旋转、剔除等数据增强），网络输入这两个图片得到embedding，要求这两个embedding具有相似性；对于所有图片产生的embedding，进行一个聚类得到原型。详细可以参考 https://arxiv.org/abs/2006.09882

​		G_e是一个content encoder，其输入为源域图片，输出为Z_c即源于的content embedding，而G_d则是把源域的content和目标域的style（这里的style是由辨别器给出的）给混合起来，得到一个输出fake，辨别器需要能辨别fake和true，这里是original的GAN流程。

​		(c)训练生成器：1个对抗损失和3个重构损失

​		生成器包括G_e和G_d两个部分，分别负责对content进行encode、和利用style\content的embedding生成图片。

​		训练主要围绕G_d，首先一条branch是original的GAN：源域图片输入G_e得到content embedding，结合目标域的style embedding一起送给G_d，生成图片以欺骗D。

​		为了更好地训练G_d利用style\content embedding的能力，增加了三个类似于自编码器的重构监督函数：

​		1.用G_e对生成图片进行encode，要求生成的content embedding与输入的相似。这里和cycleGAN有些相似。

​		2.用D分别对生成图片进行encode，要求生成的style embedding与输入的相似。

​		3.另外还有一条支路是让G_d使用D对源域图片给出的style embedding进行生成原图的操作，这里也是类似于自编码器的重构任务。

​		总的来说，这篇论文主要是把style和content的embedding给显式地解耦了，并且把辨别器当作一个encoder的思路在NICE-GAN有详细讨论：https://arxiv.org/abs/2003.00273。

