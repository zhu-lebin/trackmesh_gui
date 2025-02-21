import sys
import json
import yaml
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QOpenGLVersionProfile, QPixmap, QSurfaceFormat, QImage, QPainter
from PyQt5.QtWidgets import (QApplication, QWidget, QOpenGLWidget,
                             QHBoxLayout, QVBoxLayout, QPushButton, 
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QFormLayout, QLabel, QComboBox, QGraphicsView, QGraphicsScene, QLineEdit)
from scipy.spatial.transform import Rotation as R
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


class MyGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(MyGLWidget, self).__init__(parent)

        # 设置表面格式启用Alpha通道
        fmt = QSurfaceFormat()
        fmt.setAlphaBufferSize(8)  # 启用8位Alpha通道
        fmt.setSamples(4)          # 可选：多重采样抗锯齿
        self.setFormat(fmt)

        self.vertices = []    # 顶点数据 [x, y, z, r, g, b]
        self.faces = []       # 面索引

        # 手动变换参数（共享的变换参数）
        self.rotation = [0, 0, 0.0]  # X, Y, Z旋转角度
        self.translation = [0.0, 0.0, 0.0] # X, Y, Z平移量
        self.scale = [1.0, 1.0, 1.0]       # X, Y, Z缩放比例

        # 当前相机参数，默认为 None
        self.camera_param = None

        # 背景图片
        self.background_image = None
        #默认渲染视口大小
        self.width_value = 1920  # 默认宽度
        self.height_value = 1080 # 默认高度

    def set_mesh(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.update()

    def set_camera_param(self, param):
        """设置当前相机参数，激活透视投影及视图变换"""
        self.camera_param = param
        self.update()  # 通知重绘

    def set_background_image(self, image_path):
        """设置背景图片"""
        self.background_image = QImage(image_path)
        self.update()

    def initializeGL(self):
        version_profile = QOpenGLVersionProfile()
        version_profile.setVersion(2, 0)
        self.gl = self.context().versionFunctions(version_profile)
        self.gl.initializeOpenGLFunctions()

        # 设置透明清除颜色（RGBA中A=0）
        self.gl.glClearColor(0.0, 0.0, 0.0, 0.0)  # 全透明背景

        # 配置混合模式
        self.gl.glEnable(self.gl.GL_BLEND)
        self.gl.glBlendFunc(self.gl.GL_SRC_ALPHA, self.gl.GL_ONE_MINUS_SRC_ALPHA)

        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glDisable(self.gl.GL_LIGHTING)
        self.gl.glEnable(self.gl.GL_COLOR_MATERIAL)
    #debug:resize只有在初始化以及窗口大小改变时才会调用；width, height似乎是窗口大小而非渲染视口大小
    def resizeGL(self, width, height):
        #print(width, height)
        self.gl.glViewport(0, 0, width, height)
        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()

        if self.camera_param is not None:
            # 获取相机内参
            K = np.array(self.camera_param['K'], dtype=np.float32).reshape((3, 3))
            #fx水平焦距，fy垂直焦距，
            #cx水平主点坐标（图像宽度的一半）和cy垂直主点坐标（图像高度的一半）
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            n=0.1
            f=1000.0
            # 设置透视投影矩阵
            left   = -cx * n / fx
            right  = (self.width_value - cx) * n / fx
            bottom = -(self.height_value - cy) * n / fy
            top    = cy * n / fy
            self.gl.glFrustum(left, right, bottom, top, n, f)

        else:
            side = min(width, height)
            if side < 0:
                return
            self.gl.glOrtho(-1.5, 1.5, -1.5, 1.5, -10, 10)

        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)

    def draw_axes(self):
        # 设置坐标轴的颜色和宽度
        glLineWidth(2)
        
        # 绘制 X 轴 (红色)
        glColor3f(1.0, 0.0, 0.0)  # 红色
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)  # X轴到(1, 0, 0)
        glEnd()
        
        # 绘制 Y 轴 (绿色)
        glColor3f(0.0, 1.0, 0.0)  # 绿色
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)  # Y轴到(0, 1, 0)
        glEnd()

        # 绘制 Z 轴 (蓝色)
        glColor3f(0.0, 0.0, 1.0)  # 蓝色
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)  # Z轴到(0, 0, 1)
        glEnd()



    def draw_axes_with_camera(self):

        #坐标轴旋转回到世界坐标轴,注意旋转顺序必须倒过来因为旋转操作没有交换性
        self.gl.glRotatef(-self.rotation[2], 0, 0, 1)
        self.gl.glRotatef(-self.rotation[1], 0, 1, 0)
        self.gl.glRotatef(-self.rotation[0], 1, 0, 0)
        self.gl.glScalef(0.5, 0.5, 0.5)  # 坐标轴的缩放（固定大小）

        # 画坐标轴
        self.draw_axes()

    def paintGL(self):
        # 清除颜色缓冲区和深度缓冲区
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)
        self.gl.glLoadIdentity()

        # 绘制背景图片
        if self.background_image:
            self.draw_background()

        if self.camera_param is not None:
            #print('设置外参')
            # 使用相机外参构建视图矩阵
            #print(self.camera_param)
            R = np.array(self.camera_param['R'], dtype=np.float32).reshape((3, 3))
            t = np.array(self.camera_param['T'], dtype=np.float32)
            R_T = R.T
            t_new = -np.dot(R_T, t)
            view_matrix = np.eye(4, dtype=np.float32)
            view_matrix[:3, :3] = R_T
            view_matrix[:3, 3] = t_new
            matrix = view_matrix.T.flatten().astype(np.float32).tolist()
            self.gl.glLoadMatrixf(matrix)

            # 获取相机内参
            K = np.array(self.camera_param['K'], dtype=np.float32).reshape((3, 3))
            #fx水平焦距，fy垂直焦距，
            #cx水平主点坐标（图像宽度的一半）和cy垂直主点坐标（图像高度的一半）
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            n=0.1
            f=1000.0
            # 设置透视投影矩阵
            left   = -cx * n / fx
            right  = (self.width_value - cx) * n / fx
            bottom = -(self.height_value - cy) * n / fy
            top    = cy * n / fy
            self.gl.glFrustum(left, right, bottom, top, n, f)

        # 使用共享变换参数（平移 -> 旋转 -> 缩放）
        self.gl.glTranslatef(*self.translation)
        self.gl.glRotatef(self.rotation[0], 1, 0, 0)
        self.gl.glRotatef(self.rotation[1], 0, 1, 0)
        self.gl.glRotatef(self.rotation[2], 0, 0, 1)
        self.gl.glScalef(*self.scale)

        # 启用混合以实现透明效果
        self.gl.glEnable(self.gl.GL_BLEND)
        self.gl.glBlendFunc(self.gl.GL_SRC_ALPHA, self.gl.GL_ONE_MINUS_SRC_ALPHA)

        # 绘制网格
        if self.vertices and self.faces:
            self.gl.glBegin(self.gl.GL_TRIANGLES)
            for face in self.faces:
                for vertex_index in face:
                    if vertex_index >= len(self.vertices):
                        continue
                    vertex = self.vertices[vertex_index]
                    # 设置颜色时添加透明度（Alpha 值）半透明
                    self.gl.glColor4f(vertex[3], vertex[4], vertex[5], 0.5)  # 0.5 是透明度
                    self.gl.glVertex3fv(vertex[0:3])
            self.gl.glEnd()

        # 绘制坐标轴
        self.draw_axes_with_camera()        
        # 禁用混合
        self.gl.glDisable(self.gl.GL_BLEND)

    def draw_background(self):
        """绘制背景图片"""
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.background_image)
        painter.end()

    def set_rotation(self, x=None, y=None, z=None):
        if x is not None: self.rotation[0] = x
        if y is not None: self.rotation[1] = y
        if z is not None: self.rotation[2] = z
        self.update()

    def set_translation(self, x=None, y=None, z=None):
        if x is not None: self.translation[0] = x
        if y is not None: self.translation[1] = y
        if z is not None: self.translation[2] = z
        self.update()

    def set_scale(self, x=None, y=None, z=None):
        if x is not None: self.scale[0] = x
        if y is not None: self.scale[1] = y
        if z is not None: self.scale[2] = z
        self.update()

    def set_height(self, height):
        self.height_value = height
    
    def set_weight(self, weight):
        self.width_value = weight

class MeshViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.camera_params = []  # 存储加载的所有相机参数
        self.initUI()

    def initUI(self):
        self.setWindowTitle('OBJ Viewer with Controls')
        self.setGeometry(300, 300, 1600, 900)  # 增大窗口尺寸

        # 创建OpenGL部件
        self.glWidget1 = MyGLWidget()
        self.glWidget2 = MyGLWidget()
        self.glWidget3 = MyGLWidget()
        self.glWidget4 = MyGLWidget()
        self.glWidget5 = MyGLWidget()
        self.glWidget6 = MyGLWidget()

        # 创建控制面板
        controlPanel = self.createControlPanel()


        # 使用水平布局
        mainLayout = QHBoxLayout()  
        # 左侧布局：渲染区域
        glLayout = QVBoxLayout()    
        glLayout.addWidget(self.glWidget1, 1)
        glLayout.addWidget(self.glWidget2, 1)
        glLayout.addWidget(self.glWidget3, 1)
        # 中间布局：渲染区域   
        midLayout = QVBoxLayout()
        midLayout.addWidget(self.glWidget4, 1)
        midLayout.addWidget(self.glWidget5, 1)
        midLayout.addWidget(self.glWidget6, 1)    
        # 右侧布局：控制面板
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(controlPanel)
        rightLayout.addStretch()

        # 将左侧和右侧布局添加到主布局
        mainLayout.addLayout(glLayout, 2)  # 左侧占 2 份
        mainLayout.addLayout(midLayout, 2)  # 中间占 2 份
        mainLayout.addLayout(rightLayout, 2)  # 右侧占 2 份
        self.setLayout(mainLayout)

    def createControlPanel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        # OBJ文件操作按钮
        btnOpen = QPushButton('Open OBJ')
        btnOpen.clicked.connect(self.openOBJ)
        layout.addWidget(btnOpen)

        # 相机参数文件加载按钮
        btnLoadCamera = QPushButton("Select Camera Extriyml")
        btnLoadCamera.clicked.connect(self.openCameraEXTRIYML)
        layout.addWidget(btnLoadCamera)

        btnLoadCamera = QPushButton("Select Camera Intriyml")
        btnLoadCamera.clicked.connect(self.openCameraINRIYML)
        layout.addWidget(btnLoadCamera)

        btnLoadCamera = QPushButton("Load Camera Parameters")
        btnLoadCamera.clicked.connect(self.load_camera_parameters)
        layout.addWidget(btnLoadCamera)

        # 创建标签和输入框
        self.weight_label = QLabel('Weight:')
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText('Enter your weight')

        self.height_label = QLabel('Height:')
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText('Enter your height')

        # 创建提交按钮
        self.submit_button = QPushButton('Submit weight and height（only active after loading camera params）')
        self.submit_button.clicked.connect(self.on_submit)

        # 将控件添加到布局中
        layout.addWidget(self.weight_label)
        layout.addWidget(self.weight_input)
        layout.addWidget(self.height_label)
        layout.addWidget(self.height_input)
        layout.addWidget(self.submit_button)

        # 本地图片加载按钮
        btnLoadImage1 = QPushButton("Load Background Image 1")
        btnLoadImage1.clicked.connect(self.loadBackgroundImage1)
        layout.addWidget(btnLoadImage1)

        btnLoadImage2 = QPushButton("Load Background Image 2")
        btnLoadImage2.clicked.connect(self.loadBackgroundImage2)
        layout.addWidget(btnLoadImage2)

        btnLoadImage3 = QPushButton("Load Background Image 3")
        btnLoadImage3.clicked.connect(self.loadBackgroundImage3)
        layout.addWidget(btnLoadImage3)

        btnLoadImage4 = QPushButton("Load Background Image 4")
        btnLoadImage4.clicked.connect(self.loadBackgroundImage4)
        layout.addWidget(btnLoadImage4)

        btnLoadImage5 = QPushButton("Load Background Image 5") 
        btnLoadImage5.clicked.connect(self.loadBackgroundImage5)
        layout.addWidget(btnLoadImage5)

        btnLoadImage6 = QPushButton("Load Background Image 6")
        btnLoadImage6.clicked.connect(self.loadBackgroundImage6)
        layout.addWidget(btnLoadImage6)

        # 相机选择下拉框
        layout.addWidget(QLabel("Select Camera for Widget 1:"))
        self.cameraCombo1 = QComboBox()
        self.cameraCombo1.currentIndexChanged.connect(self.cameraChanged1)
        layout.addWidget(self.cameraCombo1)

        layout.addWidget(QLabel("Select Camera for Widget 2:"))
        self.cameraCombo2 = QComboBox()
        self.cameraCombo2.currentIndexChanged.connect(self.cameraChanged2)
        layout.addWidget(self.cameraCombo2)

        layout.addWidget(QLabel("Select Camera for Widget 3:"))
        self.cameraCombo3 = QComboBox()
        self.cameraCombo3.currentIndexChanged.connect(self.cameraChanged3)
        layout.addWidget(self.cameraCombo3)

        layout.addWidget(QLabel("Select Camera for Widget 4:"))
        self.cameraCombo4 = QComboBox()
        self.cameraCombo4.currentIndexChanged.connect(self.cameraChanged4)
        layout.addWidget(self.cameraCombo4)

        layout.addWidget(QLabel("Select Camera for Widget 5:"))
        self.cameraCombo5 = QComboBox()
        self.cameraCombo5.currentIndexChanged.connect(self.cameraChanged5)
        layout.addWidget(self.cameraCombo5)

        layout.addWidget(QLabel("Select Camera for Widget 6:"))
        self.cameraCombo6 = QComboBox()
        self.cameraCombo6.currentIndexChanged.connect(self.cameraChanged6)
        layout.addWidget(self.cameraCombo6)       

        # 添加平移、旋转、缩放控制
        layout.addWidget(self.createTranslationControls())
        layout.addWidget(self.createRotationControls())
        layout.addWidget(self.createScaleControls())

        # 导出参数按钮
        self.exportButton = QPushButton('导出参数')
        self.exportButton.clicked.connect(self.export_parameters)
        layout.addWidget(self.exportButton)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def on_submit(self):
        weight = int(self.weight_input.text())
        height = int(self.height_input.text())
        self.glWidget1.set_weight(weight)
        self.glWidget1.set_height(height)
        self.glWidget2.set_weight(weight)
        self.glWidget2.set_height(height)
        self.glWidget3.set_weight(weight)
        self.glWidget3.set_height(height)
        self.glWidget4.set_weight(weight)
        self.glWidget4.set_height(height)
        self.glWidget5.set_weight(weight)
        self.glWidget5.set_height(height)
        self.glWidget6.set_weight(weight)
        self.glWidget6.set_height(height)
        self.glWidget1.update()
        self.glWidget2.update()
        self.glWidget3.update()
        self.glWidget4.update()
        self.glWidget5.update()
        self.glWidget6.update()

    def export_parameters(self):
        # 获取旋转、平移和缩放参数
        rotation_degrees = self.get_rotation()  # 假设此方法返回旋转角度列表
        translation = self.get_translation()    # 假设此方法返回平移向量
        scale = self.get_scale()                # 假设此方法返回缩放因子

        # 将旋转角度转换为四元数
        rotation_radians = np.radians(rotation_degrees)
        rotation_matrix = R.from_euler('xyz', rotation_radians).as_matrix()
        rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()

        # 创建参数字典
        transform_params = {
            'rotation': rotation_quaternion.tolist(),
            'translation': translation,
            'scale': scale
        }

        # 保存为JSON文件
        filename, _ = QFileDialog.getSaveFileName(self, "保存参数文件", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f:
                json.dump(transform_params, f)

    def get_rotation(self):
        # 返回当前旋转角度列表 [x, y, z]
        return [self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value()]

    def get_translation(self):
        # 返回当前平移向量 [x, y, z]
        return [self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value()]

    def get_scale(self):
        # 返回当前缩放因子
        return self.scaleXSpin.value()
    #TODO:修改ui变化条
    def createTranslationControls(self):
        group = QGroupBox("Translation Controls")
        form = QFormLayout()

        # X 平移
        self.transXSpin = QDoubleSpinBox()
        self.transXSpin.setRange(-30.0, 30.0)
        self.transXSpin.setSingleStep(0.1)
        self.transXSpin.setValue(0.0)
        self.transXSpin.valueChanged.connect(self.translationChanged)
        form.addRow("X(red axis) Translation", self.transXSpin)

        # Y 平移
        self.transYSpin = QDoubleSpinBox()
        self.transYSpin.setRange(-30.0, 30.0)
        self.transYSpin.setSingleStep(0.1)
        self.transYSpin.setValue(0.0)
        self.transYSpin.valueChanged.connect(self.translationChanged)
        form.addRow("Y(green axis) Translation", self.transYSpin)

        # Z 平移
        self.transZSpin = QDoubleSpinBox()
        self.transZSpin.setRange(-30.0, 30.0)
        self.transZSpin.setSingleStep(0.1)
        self.transZSpin.setValue(0.0)
        self.transZSpin.valueChanged.connect(self.translationChanged)
        form.addRow("Z(blue axis) Translation", self.transZSpin)

        group.setLayout(form)
        return group

    def createRotationControls(self):
        group = QGroupBox("Rotation Controls")
        form = QFormLayout()

        # X 旋转
        self.rotXSpin = QDoubleSpinBox()
        self.rotXSpin.setRange(-180.0, 180.0)
        self.rotXSpin.setSingleStep(5.0)
        self.rotXSpin.setValue(0)
        self.rotXSpin.valueChanged.connect(self.rotationChanged)
        form.addRow("X(red axis) Rotation", self.rotXSpin)

        # Y 旋转
        self.rotYSpin = QDoubleSpinBox()
        self.rotYSpin.setRange(-180.0, 180.0)
        self.rotYSpin.setSingleStep(5.0)
        self.rotYSpin.setValue(0)
        self.rotYSpin.valueChanged.connect(self.rotationChanged)
        form.addRow("Y(green axis) Rotation", self.rotYSpin)

        # Z 旋转
        self.rotZSpin = QDoubleSpinBox()
        self.rotZSpin.setRange(-180.0, 180.0)
        self.rotZSpin.setSingleStep(5.0)
        self.rotZSpin.setValue(0.0)
        self.rotZSpin.valueChanged.connect(self.rotationChanged)
        form.addRow("Z(blue axis) Rotation", self.rotZSpin)

        group.setLayout(form)
        return group

    def createScaleControls(self):
        group = QGroupBox("Scale Controls")
        form = QFormLayout()

        # 缩放
        self.scaleXSpin = QDoubleSpinBox()
        self.scaleXSpin.setRange(0.01, 20.01)
        self.scaleXSpin.setSingleStep(0.2)
        self.scaleXSpin.setValue(1.0)
        self.scaleXSpin.valueChanged.connect(self.scaleChanged)
        form.addRow("Scale", self.scaleXSpin)

        group.setLayout(form)
        return group

    def openOBJ(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open OBJ File", "", "OBJ Files (*.obj)")
        if not filename:
            return

        vertices, faces = self.parseOBJ(filename)
        if vertices and faces:
            self.glWidget1.set_mesh(vertices, faces)
            self.glWidget2.set_mesh(vertices, faces)
            self.glWidget3.set_mesh(vertices, faces)
            self.glWidget4.set_mesh(vertices, faces)
            self.glWidget5.set_mesh(vertices, faces)
            self.glWidget6.set_mesh(vertices, faces)
 
    #TODO
    def read_intrinsics(self, yaml_path):
    # 打开YAML文件
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        
        # 读取所有name
        names_node = fs.getNode("names")
        names = []
        for i in range(names_node.size()):
            names.append(int(names_node.at(i).string()))

        # 为每个name读取对应参数
        params_dict = {}
        for name in names:
            entry = {
                "K":fs.getNode(f"K_{name}").mat().reshape((3, 3)),
                "dist":fs.getNode(f"dist_{name}").mat().flatten()
            }
            params_dict[name] = entry      
        fs.release()  
        #print(params_dict)     
        return params_dict

    def read_extrinsics(self, yaml_path):
    # 打开YAML文件
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        
        # 读取所有name
        names_node = fs.getNode("names")
        names = []
        for i in range(names_node.size()):
            names.append(int(names_node.at(i).string()))

        # 为每个name读取对应参数
        params_dict = {}
        for name in names:
            entry = {
                "R": fs.getNode(f"Rot_{name}").mat(),
                "T": fs.getNode(f"T_{name}").mat().flatten()
            }
            params_dict[name] = entry      
        fs.release()   
        #print(params_dict)    
        return params_dict

    def load_camera_parameters(self):
        intrinsics = self.read_intrinsics(self.intri_filename)
        extrinsics = self.read_extrinsics(self.extri_filename)
        camera_params = []
        #TODO:检查重合name
        for name in intrinsics:
            if name in extrinsics:
                camera_params.append({"id":name,**intrinsics[name], **extrinsics.get(name, {})})
        #print(camera_params)
        self.camera_params = camera_params if isinstance(camera_params, list) else [camera_params]
        self.updateCameraCombos()



    def openCameraINRIYML(self):
        intri_filename, _ = QFileDialog.getOpenFileName(self, "Open Intrinsics YAML", "", "YAML Files (*.yml)")
        if not intri_filename:
            return
        else:
            self.intri_filename = intri_filename

    def openCameraEXTRIYML(self):
        extri_filename, _ = QFileDialog.getOpenFileName(self, "Open Extrinsics YAML", "", "YAML Files (*.yml)")
        if not extri_filename:
            return
        else:
            self.extri_filename = extri_filename


    def loadBackgroundImage1(self):
        """加载背景图片"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Background Image 1", "", "Image Files (*.png *.jpg *.bmp)")
        if not filename:
            return

        # 为每个 OpenGL 部件设置背景图片
        self.glWidget1.set_background_image(filename)

    def loadBackgroundImage2(self):
        """加载背景图片"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Background Image 2", "", "Image Files (*.png *.jpg *.bmp)")
        if not filename:
            return

        # 为每个 OpenGL 部件设置背景图片
        self.glWidget2.set_background_image(filename)

    def loadBackgroundImage3(self):
        """加载背景图片"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Background Image 3", "", "Image Files (*.png *.jpg *.bmp)")
        if not filename:
            return

        # 为每个 OpenGL 部件设置背景图片
        self.glWidget3.set_background_image(filename)

    def loadBackgroundImage4(self):
        """加载背景图片"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Background Image 4", "", "Image Files (*.png *.jpg *.bmp)")
        if not filename:
            return

        # 为每个 OpenGL 部件设置背景图片
        self.glWidget4.set_background_image(filename)

    def loadBackgroundImage5(self):
        """加载背景图片"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Background Image 5", "", "Image Files (*.png *.jpg *.bmp)")
        if not filename:
            return

        # 为每个 OpenGL 部件设置背景图片
        self.glWidget5.set_background_image(filename)
    
    def loadBackgroundImage6(self):
        """加载背景图片"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Background Image 6", "", "Image Files (*.png *.jpg *.bmp)")
        if not filename:
            return

        # 为每个 OpenGL 部件设置背景图片
        self.glWidget6.set_background_image(filename)
#TODO
    def updateCameraCombos(self):
        self.cameraCombo1.clear()
        self.cameraCombo2.clear()
        self.cameraCombo3.clear()
        self.cameraCombo4.clear()
        self.cameraCombo5.clear()
        self.cameraCombo6.clear()
        for cam in self.camera_params:
            text = f"ID {cam.get('id', '')}"
            self.cameraCombo1.addItem(text, cam)
            self.cameraCombo2.addItem(text, cam)
            self.cameraCombo3.addItem(text, cam)
            self.cameraCombo4.addItem(text, cam)
            self.cameraCombo5.addItem(text, cam)
            self.cameraCombo6.addItem(text, cam)

    def cameraChanged1(self, index):
        if index < 0 or index >= len(self.camera_params):
            return
        cam = self.cameraCombo1.itemData(index)
        self.glWidget1.set_camera_param(cam)

    def cameraChanged2(self, index):
        if index < 0 or index >= len(self.camera_params):
            return
        cam = self.cameraCombo2.itemData(index)
        self.glWidget2.set_camera_param(cam)

    def cameraChanged3(self, index):
        if index < 0 or index >= len(self.camera_params):
            return
        cam = self.cameraCombo3.itemData(index)
        self.glWidget3.set_camera_param(cam)

    def cameraChanged4(self, index):
        if index < 0 or index >= len(self.camera_params):
            return
        cam = self.cameraCombo4.itemData(index)
        self.glWidget4.set_camera_param(cam)

    def cameraChanged5(self, index):
        if index < 0 or index >= len(self.camera_params):
            return
        cam = self.cameraCombo5.itemData(index)
        self.glWidget5.set_camera_param(cam)

    def cameraChanged6(self, index):
        if index < 0 or index >= len(self.camera_params):
            return
        cam = self.cameraCombo6.itemData(index)
        self.glWidget6.set_camera_param(cam)



    def translationChanged(self, value):
        self.glWidget1.set_translation(self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value())
        self.glWidget2.set_translation(self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value())
        self.glWidget3.set_translation(self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value())
        self.glWidget4.set_translation(self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value())
        self.glWidget5.set_translation(self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value())
        self.glWidget6.set_translation(self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value())
    def rotationChanged(self, value):
        self.glWidget1.set_rotation(self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value())
        self.glWidget2.set_rotation(self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value())
        self.glWidget3.set_rotation(self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value())
        self.glWidget4.set_rotation(self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value())
        self.glWidget5.set_rotation(self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value())
        self.glWidget6.set_rotation(self.rotXSpin.value(), self.rotYSpin.value(), self.rotZSpin.value())
    def scaleChanged(self, value):
        self.glWidget1.set_scale(self.scaleXSpin.value(), self.scaleXSpin.value(), self.scaleXSpin.value())
        self.glWidget2.set_scale(self.scaleXSpin.value(), self.scaleXSpin.value(), self.scaleXSpin.value())
        self.glWidget3.set_scale(self.scaleXSpin.value(), self.scaleXSpin.value(), self.scaleXSpin.value())
        self.glWidget4.set_scale(self.scaleXSpin.value(), self.scaleXSpin.value(), self.scaleXSpin.value())
        self.glWidget5.set_scale(self.scaleXSpin.value(), self.scaleXSpin.value(), self.scaleXSpin.value())
        self.glWidget6.set_scale(self.scaleXSpin.value(), self.scaleXSpin.value(), self.scaleXSpin.value())
    def parseOBJ(self, filename):
        vertices = []
        faces = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v' and len(parts) >= 7:
                        coords = list(map(float, parts[1:7]))
                        if any(c > 1.0 for c in coords[3:6]):
                            coords[3:6] = [c/255.0 for c in coords[3:6]]
                        vertices.append(coords)
                    elif parts[0] == 'f':
                        face = []
                        for v in parts[1:4]:
                            index = int(v.split('/')[0]) - 1
                            face.append(index)
                        faces.append(face)
            return vertices, faces
        except Exception as e:
            print(f"Error reading OBJ file: {e}")
            return [], []

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MeshViewer()
    viewer.show()
    sys.exit(app.exec())