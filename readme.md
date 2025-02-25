## 使用说明
- 先读取obj文件，目前仅支持三角网格
- 读取外参yml和内参yml文件，点击导入相机参数即可选择相机
- 渲染分辨率默认1920*1080，可在height和weight输入框修改后提交，gui窗口可拉伸放大，渲染结果和图片会随着一起拉伸
- 选择背景图片，调整平移旋转缩放参数使得渲染结果和图片对齐，可修改代码里createTranslationControls函数与createRotationControls函数的参数来调整参数可变范围以及每步大小
- 其他:渲染结果目前是半透明的，不过可能不如不透明好用，考虑改掉；添加了一个随着物体平移但不旋转的世界坐标轴，方便调整旋转角度。

## 更新

- trackmesh_yml.py用于yaml格式，并修改了之前的渲染分辨率的bug，虽然我在这方面还是不太确定是否正确。
需要安装opencv

2.25
- 修改了相机内参外参导入之后设置视图矩阵和投影矩阵的bug，参考https://blog.csdn.net/yanglusheng/article/details/52268234
由于opencv和opengl的默认相机旋转朝向不同，所以要将opengl的相机先绕x轴旋转180度到opencv的相机位置，或者说将opencv的相机旋转平移参数乘一个旋转矩阵；另一个bug是opengl的矩阵用列主序存储和numpy不一样
- 添加了图像自动去畸变