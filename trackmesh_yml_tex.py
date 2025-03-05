import sys
import json
import yaml
import math
import cv2
import numpy as np
import os
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
from pyrr import Vector3, Matrix44
#TODO检查导出参数
#修改了着色器，只是用了diffuse map，代码需要优化修正,想改成test1的样子，在set_mesh里初始化网格数据的
#现在不支持无纹理的mesh了

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
        self.distortion_enabled = False  # 默认不启用去畸变

        self.diffuse_texture = None
        self.normal_texture = None
        self.ao_texture = None
        self.diffuse_map = None
        self.normal_map = None
        self.ao_map = None
        self.reflectivity = 0
        self.shader_program = None

    def set_mesh(self, vertices, faces,normals, textures, materials):
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.textures = textures
        self.materials = materials
        self.diffuse_map = materials['map_Kd'] if 'map_Kd' in materials else None
        self.normal_map = materials['norm'] if 'norm' in materials else None
        self.ao_map = materials['map_ao'] if 'map_ao' in materials else None
        self.reflectivity = materials['Pr'] if 'Pr' in materials else 0
        # if self.diffuse_map is not None:
        #     # 初始化网格数据
        #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #     glEnable(GL_TEXTURE_2D)
        #     glEnable(GL_DEPTH_TEST)
        #     self.init_mesh_data()
        self.update()

    def set_camera_param(self, param):
        """设置当前相机参数，激活透视投影及视图变换"""
        self.camera_param = param
        self.update()  # 通知重绘

    def set_background_image(self, image_path):
        """设置背景图片"""
        self.background_image = QImage(image_path)
        self.update()

    #     return program
    def initializeGL(self):
        # 设置透明清除颜色（RGBA中A=0）
        glClearColor(0.0, 0.0, 0.0, 0.0)  # 全透明背景

        # 配置混合模式
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)

        # 初始化着色器
        self.init_shaders()

    def load_texture_qimage(self, image_path):
        # 读取图像
        image = QImage(image_path)
        if image.isNull():
            raise ValueError("Failed to load texture image!")

        # 转换为OpenGL支持的格式（JPG需要转换为RGBA）
        image = image.convertToFormat(QImage.Format_RGBA8888)
        # 获取图像数据并转换为bytes
        image_data = image.bits().asstring(image.sizeInBytes())
        # 生成纹理ID
        texture_id = glGenTextures(1)
        texture_id = int(texture_id)  # 转换为int

        # 绑定纹理
        glBindTexture(GL_TEXTURE_2D, texture_id)

        # 上传纹理数据
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(),
            0, GL_RGBA, GL_UNSIGNED_BYTE, image_data
        )

        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        return texture_id

    def init_shaders(self):
        # 顶点着色器
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec2 aTexCoord;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        out vec2 TexCoord;
        void main() {
            vec4 world_position = model * vec4(aPos, 1.0);
            gl_Position = projection * view * world_position;
            TexCoord = aTexCoord;
        }
        """

        # 片段着色器
        fragment_shader = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D textureSampler;
        void main() {
            FragColor = texture(textureSampler, TexCoord);
            
        }
        """
        #FragColor = vec4(TexCoord, 0.0, 1.0);
        #FragColor = texture(textureSampler, TexCoord);
        # 创建着色器程序
        self.shader_program = glCreateProgram()
        vertex_shader_id = self.compile_shader(GL_VERTEX_SHADER, vertex_shader)
        fragment_shader_id = self.compile_shader(GL_FRAGMENT_SHADER, fragment_shader)
        glAttachShader(self.shader_program, vertex_shader_id)
        glAttachShader(self.shader_program, fragment_shader_id)
        glLinkProgram(self.shader_program)

        # 检查链接状态
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(self.shader_program))

    def compile_shader(self, shader_type, source):
        shader_id = glCreateShader(shader_type)
        glShaderSource(shader_id, source)
        glCompileShader(shader_id)
        if not glGetShaderiv(shader_id, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader_id))
        return shader_id

    #debug:resize只有在初始化以及窗口大小改变时才会调用；width, height似乎是窗口大小而非渲染视口大小
    def resizeGL(self, width, height):
        #print(width, height)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.camera_param is not None:
            K = np.array(self.camera_param['K'], dtype=np.float32).reshape((3, 3))
            #fx水平焦距，fy垂直焦距，
            #cx水平主点坐标（图像宽度的一半）和cy垂直主点坐标（图像高度的一半）
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            n=0.1
            f=30
            w=self.width_value
            h=self.height_value
            left = -cx * n / fx
            right = (w - cx) * n / fx
            bottom = -(h - cy) * n / fy
            top = cy * n / fy
            # 确保top > bottom
            top, bottom = max(top, bottom), min(top, bottom)

            # 计算投影矩阵参数
            rl = right - left
            tb = top - bottom

            # 构建投影矩阵（行主序）
            proj_matrix = np.array([
                [2*n/rl, 0,        (right + left)/rl, 0],
                [0,      2*n/tb,  (top + bottom)/tb,  0],
                [0,      0,       -(f + n)/(f - n),  -2*f*n/(f - n)],
                [0,      0,       -1,                0]
            ], dtype=np.float32)

            glMatrixMode(GL_PROJECTION)  # 切换到投影矩阵模式
            glLoadIdentity()
            #注意列主序存储
            glLoadMatrixf(proj_matrix.T.flatten().astype(np.float32).tolist())
            glMatrixMode(GL_MODELVIEW)   # 切换到模型视图矩阵模式

        else:
            side = min(width, height)
            if side < 0:
                return
            glOrtho(-4, 4, -4, 4, -10, 10)

        glMatrixMode(GL_MODELVIEW)

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

        # #坐标轴旋转回到世界坐标轴,注意旋转顺序必须倒过来因为旋转操作没有交换性
        glRotatef(-self.rotation[2], 0, 0, 1)
        glRotatef(-self.rotation[1], 0, 1, 0)
        glRotatef(-self.rotation[0], 1, 0, 0)
        #平移回原点     
        #glTranslatef(-self.translation[0], -self.translation[1], -self.translation[2])
        glScalef(0.5, 0.5, 0.5)  # 坐标轴的缩放（固定大小）
        # 画坐标轴
        self.draw_axes()


    def init_mesh_data(self):
        # 将顶点坐标和纹理坐标打包为交错数组
        self.vertex_data = []
        self.indices = []
        #print(self.faces)
        for face in self.faces:
            self.indices.extend(face)
        for i in range(len(self.vertices)):
            #纹理坐标系与 OpenGL 的默认坐标系 不同,OpenGL 的纹理坐标系原点 (0, 0) 位于 左下角,许多图像格式（如 PNG、JPEG）坐标系原点 (0, 0) 位于 左上角
            self.vertex_data.extend(self.vertices[i])
            self.vertex_data.extend([self.textures[i][0]])
            self.vertex_data.extend([1-self.textures[i][1]])

        # 将数据转换为 numpy 数组
        self.vertex_data = np.array(self.vertex_data, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)
        # self.indices = np.roll(self.indices,1)
        #print(self.vertex_data[0:10])
        #print(self.indices[0:120])
        # 生成 VAO、VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        # 绑定 VAO
        glBindVertexArray(self.vao)

        # 上传顶点数据到 VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data.nbytes, self.vertex_data, GL_STATIC_DRAW)

        # 创建并绑定 EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # 设置顶点属性指针
        # 顶点坐标（位置 0）
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, None)
        glEnableVertexAttribArray(0)
        # 纹理坐标（位置 1）
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # 解绑 VAO
        glBindVertexArray(0)


    def get_model_matrix(self,translation, rotation_degrees, scale):

        translation_matrix = np.array([
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 旋转矩阵
        rotation_radians = np.radians(rotation_degrees)

        rotation_matrix_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rotation_radians[0]), -np.sin(rotation_radians[0]), 0],
            [0, np.sin(rotation_radians[0]), np.cos(rotation_radians[0]), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        rotation_matrix_y = np.array([
            [np.cos(rotation_radians[1]), 0, np.sin(rotation_radians[1]), 0],
            [0, 1, 0, 0],
            [-np.sin(rotation_radians[1]), 0, np.cos(rotation_radians[1]), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        rotation_matrix_z = np.array([
            [np.cos(rotation_radians[2]), -np.sin(rotation_radians[2]), 0, 0],
            [np.sin(rotation_radians[2]), np.cos(rotation_radians[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 缩放矩阵
        scale_matrix = np.array([
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 计算最终的 Model 矩阵
        model_matrix = translation_matrix @ rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x @ scale_matrix

        return model_matrix
    
    def orthographic_projection(self,left, right, bottom, top, near, far):
        """ 计算正交投影矩阵 """
        ortho_matrix = np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        return ortho_matrix

    def paintGL(self):
        # 清除颜色缓冲区和深度缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 绘制背景图片
        if self.background_image:
            self.draw_background()
        view_matrix = np.eye(4, dtype=np.float32)
        proj_matrix = np.eye(4, dtype=np.float32)
        projection = self.orthographic_projection(-4, 4, -4, 4, -10, 10)
        if self.camera_param is not None:
            # 获取相机内参
            K = np.array(self.camera_param['K'], dtype=np.float32).reshape((3, 3))
            #fx水平焦距，fy垂直焦距，
            #cx水平主点坐标（图像宽度的一半）和cy垂直主点坐标（图像高度的一半）
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            n=0.1
            f=100
            w=self.width_value
            h=self.height_value
            left = -cx * n / fx
            right = (w - cx) * n / fx
            bottom = -(h - cy) * n / fy
            top = cy * n / fy
            # 确保top > bottom
            top, bottom = max(top, bottom), min(top, bottom)

            # 计算投影矩阵参数
            rl = right - left
            tb = top - bottom

            # 构建投影矩阵（行主序）
            proj_matrix = np.array([
                [2*n/rl, 0,        (right + left)/rl, 0],
                [0,      2*n/tb,  (top + bottom)/tb,  0],
                [0,      0,       -(f + n)/(f - n),  -2*f*n/(f - n)],
                [0,      0,       -1,                0]
            ], dtype=np.float32)

            glMatrixMode(GL_PROJECTION)  # 切换到投影矩阵模式
            glLoadIdentity()
            #注意列主序存储
            glLoadMatrixf(proj_matrix.T.flatten().astype(np.float32).tolist())
            glMatrixMode(GL_MODELVIEW)   # 切换到模型视图矩阵模式
            
            # 使用相机外参构建视图矩阵
            #先旋转到opencv使用的默认相机视角
            C =    np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ], dtype=np.float32)
            R = np.array(self.camera_param['R'], dtype=np.float32).reshape((3, 3))
            #print(R)
            t = np.array(self.camera_param['T'], dtype=np.float32)
            R = np.dot(C,R)
            t = np.array([t[0],-t[1],-t[2]])
            #view_matrix = np.eye(4, dtype=np.float32)
            view_matrix[:3, :3] = R
            view_matrix[:3, 3] = t
            #由于opengl列存储所以必须加转置
            matrix = view_matrix.T.flatten().astype(np.float32).tolist()
            glLoadMatrixf(matrix)

        # 使用共享变换参数（平移 -> 旋转 -> 缩放）
        glTranslatef(*self.translation)

        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(*self.scale)

        # 启用混合以实现透明效果
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)

        if self.diffuse_map is not None:
            # 初始化网格数据
            self.init_mesh_data()
            # 绑定着色器程序
            glUseProgram(self.shader_program)
            glUniform1i(glGetUniformLocation(self.shader_program, "textureSampler"), 0)
            glActiveTexture(GL_TEXTURE0)
            #绑定纹理
            self.diffuse_texture = self.load_texture_qimage(self.diffuse_map)
            #仅绑定 color 贴图
            glBindTexture(GL_TEXTURE_2D, self.diffuse_texture)
            #

            # 设置模型、视图和投影矩阵
            model = self.get_model_matrix(self.translation, self.rotation, self.scale)
            view = view_matrix            
            projection = proj_matrix
            #GL_TRUE 表示 转置矩阵,但是不要用这个参数，我也不知道为什么用了会出错不如自己手动转置
            glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 1, GL_FALSE, model.T)
            glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 1, GL_FALSE, view.T)
            glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 1, GL_FALSE, projection.T)

            # 绑定VAO并绘制网格
            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            # 解绑
            glBindVertexArray(0)
            # self.draw_quad()
        # 解绑着色器程序
        glUseProgram(0)
        self.draw_axes_with_camera()        
        # 禁用混合
        glDisable(GL_BLEND)

    def draw_quad(self):
        vertices = [
            # 位置          # 纹理坐标
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5,  0.5, 0.0, 0.0, 1.0
        ]
        indices = [0, 1, 2, 2, 3, 0]

        # 上传顶点数据
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, (GLfloat * len(vertices))(*vertices), GL_STATIC_DRAW)

        # 设置顶点属性指针
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # 绘制
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)

        # 解绑
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)


    def draw_background(self):
        """绘制背景图片"""
        painter = QPainter(self)
        if self.camera_param is not None and self.distortion_enabled:
            # 获取相机内参
            K = np.array(self.camera_param['K'], dtype=np.float32).reshape((3, 3))
            distCoeffs = np.array(self.camera_param['dist'], dtype=np.float32)

            width = self.background_image.width()
            height = self.background_image.height()
            ptr = self.background_image.bits()
            ptr.setsize(self.background_image.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # 4 表示 RGBA 格式
            arr = arr[..., :3]  # 只取 RGB 通道
            image = cv2.undistort(arr, K, distCoeffs)
            image = np.uint8(np.clip(image, 0, 255))
            # 转换为 RGB 格式 OpenCV 默认使用 BGR 格式存储图像，而 Qt 中的 QImage 默认使用 RGB 格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转换去畸变后的 NumPy 数组为 QImage
            corrected_image = QImage(image_rgb.data, width, height, QImage.Format_RGB888)
            painter.drawImage(self.rect(), corrected_image)
        else:
            painter.drawImage(self.rect(), self.background_image)
        painter.end()


    def set_distortion_enabled(self, enabled):
        self.distortion_enabled = enabled
        self.update()
 
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
        self.current_width = 1920
        self.current_height = 1080
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
        self.width_label = QLabel('Width:')
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText('Enter your width')
        self.width_input.setText(str(self.current_width))  # 设置当前宽度值

        self.height_label = QLabel('Height:')
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText('Enter your height')
        self.height_input.setText(str(self.current_height))  # 设置当前高度值

        # 创建提交按钮
        self.submit_button = QPushButton('Submit weight and height（only active after loading camera params）')
        self.submit_button.clicked.connect(self.on_submit)

        # 将控件添加到布局中
        layout.addWidget(self.width_label)
        layout.addWidget(self.width_input)
        layout.addWidget(self.height_label)
        layout.addWidget(self.height_input)
        layout.addWidget(self.submit_button)

        # 本地图片加载按钮
        image_layout1 = QHBoxLayout()
        btnLoadImage1 = QPushButton("Load Background Image 1")
        btnLoadImage1.clicked.connect(self.loadBackgroundImage1)
        image_layout1.addWidget(btnLoadImage1)
        # 去畸变按钮
        btnDistortionCorrection1 = QPushButton("Enable Distortion Correction")
        btnDistortionCorrection1.setCheckable(True)  # 设置按钮为可选中
        btnDistortionCorrection1.clicked.connect(self.toggle_distortion1)
        self.distortion_enabled1 = False  # 默认不启用去畸变
        # 将按钮添加到水平布局
        image_layout1.addWidget(btnLoadImage1)
        image_layout1.addWidget(btnDistortionCorrection1)
        layout.addLayout(image_layout1)

        image_layout2 = QHBoxLayout()
        btnLoadImage2 = QPushButton("Load Background Image 2")
        btnLoadImage2.clicked.connect(self.loadBackgroundImage2)
        image_layout2.addWidget(btnLoadImage2)
        # 去畸变按钮
        btnDistortionCorrection2 = QPushButton("Enable Distortion Correction")
        btnDistortionCorrection2.setCheckable(True)  # 设置按钮为可选中
        btnDistortionCorrection2.clicked.connect(self.toggle_distortion2)
        self.distortion_enabled2 = False  # 默认不启用去畸变
        # 将按钮添加到水平布局
        image_layout2.addWidget(btnLoadImage2)
        image_layout2.addWidget(btnDistortionCorrection2)
        layout.addLayout(image_layout2)

        image_layout3 = QHBoxLayout()
        btnLoadImage3 = QPushButton("Load Background Image 3")
        btnLoadImage3.clicked.connect(self.loadBackgroundImage3)
        image_layout3.addWidget(btnLoadImage3)
        # 去畸变按钮
        btnDistortionCorrection3 = QPushButton("Enable Distortion Correction")
        btnDistortionCorrection3.setCheckable(True)
        btnDistortionCorrection3.clicked.connect(self.toggle_distortion3)
        self.distortion_enabled3 = False
        image_layout3.addWidget(btnLoadImage3)
        image_layout3.addWidget(btnDistortionCorrection3)
        layout.addLayout(image_layout3)

        image_layout4 = QHBoxLayout()
        btnLoadImage4 = QPushButton("Load Background Image 4")
        btnLoadImage4.clicked.connect(self.loadBackgroundImage4)
        image_layout4.addWidget(btnLoadImage4)
        # 去畸变按钮
        btnDistortionCorrection4 = QPushButton("Enable Distortion Correction")
        btnDistortionCorrection4.setCheckable(True)
        btnDistortionCorrection4.clicked.connect(self.toggle_distortion4)
        self.distortion_enabled4 = False
        image_layout4.addWidget(btnLoadImage4)
        image_layout4.addWidget(btnDistortionCorrection4)
        layout.addLayout(image_layout4)

        image_layout5 = QHBoxLayout()
        btnLoadImage5 = QPushButton("Load Background Image 5")
        btnLoadImage5.clicked.connect(self.loadBackgroundImage5)
        image_layout5.addWidget(btnLoadImage5)
        # 去畸变按钮
        btnDistortionCorrection5 = QPushButton("Enable Distortion Correction")
        btnDistortionCorrection5.setCheckable(True)
        btnDistortionCorrection5.clicked.connect(self.toggle_distortion5)
        self.distortion_enabled5 = False
        image_layout5.addWidget(btnLoadImage5)
        image_layout5.addWidget(btnDistortionCorrection5)
        layout.addLayout(image_layout5)

        image_layout6 = QHBoxLayout()
        btnLoadImage6 = QPushButton("Load Background Image 6")
        btnLoadImage6.clicked.connect(self.loadBackgroundImage6)
        image_layout6.addWidget(btnLoadImage6)
        # 去畸变按钮
        btnDistortionCorrection6 = QPushButton("Enable Distortion Correction")
        btnDistortionCorrection6.setCheckable(True)
        btnDistortionCorrection6.clicked.connect(self.toggle_distortion6)
        self.distortion_enabled6 = False
        image_layout6.addWidget(btnLoadImage6)
        image_layout6.addWidget(btnDistortionCorrection6)
        layout.addLayout(image_layout6)


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
    
    def toggle_distortion1(self):
        """切换是否进行去畸变"""
        self.distortion_enabled1 = not self.distortion_enabled1
        self.glWidget1.set_distortion_enabled(self.distortion_enabled1)

    def toggle_distortion2(self):
        """切换是否进行去畸变"""
        self.distortion_enabled2 = not self.distortion_enabled2
        self.glWidget2.set_distortion_enabled(self.distortion_enabled2)
    
    def toggle_distortion3(self):
        """切换是否进行去畸变"""
        self.distortion_enabled3 = not self.distortion_enabled3
        self.glWidget3.set_distortion_enabled(self.distortion_enabled3)

    def toggle_distortion4(self):
        """切换是否进行去畸变"""
        self.distortion_enabled4 = not self.distortion_enabled4
        self.glWidget4.set_distortion_enabled(self.distortion_enabled4)

    def toggle_distortion5(self):
        """切换是否进行去畸变"""
        self.distortion_enabled5 = not self.distortion_enabled5
        self.glWidget5.set_distortion_enabled(self.distortion_enabled5)

    def toggle_distortion6(self):
        """切换是否进行去畸变"""
        self.distortion_enabled6 = not self.distortion_enabled6
        self.glWidget6.set_distortion_enabled(self.distortion_enabled6)  
    def on_submit(self):
        self.current_height = int(self.height_input.text())
        self.current_width = int(self.width_input.text())
        width = int(self.width_input.text())
        height = int(self.height_input.text())
        self.glWidget1.set_weight(width)
        self.glWidget1.set_height(height)
        self.glWidget2.set_weight(width)
        self.glWidget2.set_height(height)
        self.glWidget3.set_weight(width)
        self.glWidget3.set_height(height)
        self.glWidget4.set_weight(width)
        self.glWidget4.set_height(height)
        self.glWidget5.set_weight(width)
        self.glWidget5.set_height(height)
        self.glWidget6.set_weight(width)
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
#TODO:
    def openOBJ(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open OBJ File", "", "OBJ Files (*.obj);;All Files (*)")
        if not filename:
            return

        try:
            vertices, faces, normals, textures, materials = self.parseOBJ(filename)
            # vertices, faces, _, _, _ = self.parseOBJ(filename)
            if vertices and faces:
                # 打印 vertices 和 faces 的大小
                print(f"Number of vertices: {len(vertices)}")
                print(f"Number of faces: {len(faces)}")
                
                # 更新所有的 OpenGL 小部件
                for widget in [self.glWidget1, self.glWidget2, self.glWidget3, self.glWidget4, self.glWidget5, self.glWidget6]:
                    widget.set_mesh(vertices, faces, normals, textures, materials)
        except Exception as e:
            print(f"Error loading OBJ file: {e}")

 
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
                "R_rec":fs.getNode(f"R_{name}").mat().flatten(),
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

#TODO:修改obj文件读取  
    def parseOBJ(self, filename):
        vertices = []
        faces = []
        normals = []
        textures = []
        materials = None
        current_material = None
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if not parts:
                    continue
                #TODO:临时修改问题很多
                # 解析顶点 v [x, y, z, r, g, b] 或 v [x, y, z]
                if parts[0] == 'v':
                    if len(parts) == 4:  # 仅包含 x, y, z 坐标
                        coords = list(map(float, parts[1:4]))
                        
                        # if current_material and materials and current_material in materials:
                        #     material_color = materials[current_material].get('Kd', [1.0, 1.0, 1.0])  # 默认白色
                        #     coords.extend(material_color)  # 添加材质的颜色信息
                        # else:
                        #     coords.extend([1.0, 1.0, 1.0])  # 默认白色
                        vertices.append(coords)
                    elif len(parts) == 7:  # 颜色包含 r, g, b
                        coords = list(map(float, parts[1:7]))
                        # 如果颜色超过 [1, 1, 1] 需要归一化
                        if any(c > 1.0 for c in coords[3:6]):
                            coords[3:6] = [c/255.0 for c in coords[3:6]]
                        vertices.append(coords)

                # 解析法向量 vn [x, y, z]
                elif parts[0] == 'vn':
                    normals.append(list(map(float, parts[1:4])))

                # 解析纹理坐标 vt [u, v]
                elif parts[0] == 'vt':
                    textures.append(list(map(float, parts[1:3])))

                # 解析面 f [v1/vt1/vn1, v2/vt2/vn2, v3/vt3/vn3]
                elif parts[0] == 'f':
                    face = []
                    for v in parts[1:]:
                        vertex_data = v.split('/')
                        vertex_indices = int(vertex_data[0]) - 1  # 顶点索引
                        face.append(vertex_indices)
                    faces.append(face)

                # 解析材质库 mtllib filename.mtl
                elif parts[0] == 'mtllib':
                    # 获取obj文件所在的目录
                    obj_dir = os.path.dirname(os.path.abspath(filename))
                    mtl_filename = parts[1]
                    mtl_filepath = os.path.join(obj_dir, mtl_filename)                  
                    materials = self.parseMTL(mtl_filepath)

                # 解析使用材质 usemtl material_name
                elif parts[0] == 'usemtl':
                    current_material = parts[1]
            #不知道会不会有多个材质，不考虑了
            # print(materials)
            # print(materials['Material'])
            return vertices, faces, normals, textures, materials['Material']
        except Exception as e:
            print(f"Error reading OBJ file: {e}")
            return [], [], [], [], []

    def parseMTL(self, filename):
        materials = {}
        current_material = None
        mtl_dir = os.path.dirname(os.path.abspath(filename))
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if not parts:
                    continue

                # 解析材质名 newmtl material_name
                if parts[0] == 'newmtl':
                    current_material = parts[1]
                    materials[current_material] = {}

                # 解析漫反射颜色 Kd
                elif parts[0] == 'Kd':
                    materials[current_material]['Kd'] = list(map(float, parts[1:4]))

                # 解析环境光颜色 Ka
                elif parts[0] == 'Ka':
                    materials[current_material]['Ka'] = list(map(float, parts[1:4]))

                # 解析镜面反射颜色 Ks
                elif parts[0] == 'Ks':
                    materials[current_material]['Ks'] = list(map(float, parts[1:4]))

                # 解析纹理图像 map_Kd
                elif parts[0] == 'map_Kd':
                    materials[current_material]['map_Kd'] = os.path.normpath(os.path.join(mtl_dir,parts[1]))

                elif parts[0] == 'norm':
                    materials[current_material]['norm'] =  os.path.normpath(os.path.join(mtl_dir,parts[1]))

                elif parts[0] == 'map_ao':
                    materials[current_material]['map_ao'] =  os.path.normpath(os.path.join(mtl_dir,parts[1]))

                elif parts[0] == 'Pr':
                    materials[current_material]['Pr'] = parts[1]

            return materials
        except Exception as e:
            print(f"Error reading MTL file: {e}")
            return {}

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MeshViewer()
    viewer.show()
    sys.exit(app.exec())