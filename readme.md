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

3.5
- 添加能读取带纹理网格的trackmesh_yml_tex.py，使用了自定义shader，代码改得比较糙无法读取无纹理的网格；修正了导出四元数的错误；渲染管线不考虑法向、法向贴图

3.18
- 修改背景图片没显示的bug，通过渲染纹理的方式显示背景

5.10
- 改用pytorch3d解析obj格式文件。windows配置pytorch3d注意点：先安装pytorch(1.12.0+cu116可自行调整)，使用set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6 set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%让本地环境cuda版本与conda环境一致，然后编译安装pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"。配置完pytorch3d可能会由于多个OpenMP库报错，代码中已经添加os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"防止报错；新的blender格式的obj面数据比较特别，对应的空间坐标点、纹理坐标点、法向坐标点可以不一致如f 39488/1/1 39744/2/2 39677/3/1，代码中通过自行构造一致点列解决这一问题。

7.18
- 优化了操作界面，推荐放大全屏，用滚轮来调整数值参数，修复了读取后卡死的bug