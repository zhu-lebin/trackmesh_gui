import sys
import json
import yaml
import math
import cv2
import numpy as np
import os
#防止报错链接了多个不同来源的 OpenMP 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QOpenGLVersionProfile, QPixmap, QSurfaceFormat, QImage, QPainter
from PyQt5.QtWidgets import (QApplication, QWidget, QOpenGLWidget,QMessageBox,
                             QHBoxLayout, QVBoxLayout, QPushButton, 
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QFormLayout, QLabel, QComboBox, QGraphicsView, QGraphicsScene, QLineEdit)
from scipy.spatial.transform import Rotation as R
import torch
from pytorch3d.io import load_obj
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
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
        self.background_image_path = None
        #默认渲染视口大小
        self.width_value = 1920  # 默认宽度
        self.height_value = 1080 # 默认高度
        self.distortion_enabled = False  # 默认不启用去畸变

        self.diffuse_texture = None
        self.background_texture = None 
        # self.normal_texture = None
        # self.ao_texture = None
        self.diffuse_map = None
        # self.normal_map = None
        # self.ao_map = None
        self.reflectivity = 0
        self.shader_program = None

    def set_mesh(self, vertices, indices, material):
        self.unique_vertices = vertices
        self.unique_faces = indices
        #纹理贴图路径
        self.diffuse_map = material['map_Kd'] if material is not None else None
        #暂时不考虑法向以及其他类型贴图
        # self.normals = normals
        # self.textures = textures
        # self.normal_map = materials['norm'] if 'norm' in materials else None
        # self.ao_map = materials['map_ao'] if 'map_ao' in materials else None
        # self.reflectivity = materials['Pr'] if 'Pr' in materials else 0
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
        self.background_image_path = image_path
        self.load_background_texture()
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

    #TODO兼容性，png和jpg
    def load_texture_qimage(self, image_path):
        # 读取图像
        image = QImage(image_path)
        if image.isNull():
            raise ValueError(f"Failed to load texture image: {image_path}")

        # 判断是否有 alpha 通道
        if image.hasAlphaChannel():
            image = image.convertToFormat(QImage.Format_RGBA8888)
            gl_format = GL_RGBA
        else:
            image = image.convertToFormat(QImage.Format_RGB888)
            gl_format = GL_RGB

        # 获取图像数据并转换为 bytes
        width = image.width()
        height = image.height()
        image_data = image.bits().asstring(image.sizeInBytes())

        # 生成纹理 ID
        texture_id = glGenTextures(1)
        texture_id = int(texture_id)  # 转换为 int，避免类型冲突

        # 绑定纹理并上传数据
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, gl_format, width, height, 0, gl_format, GL_UNSIGNED_BYTE, image_data)

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
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D) 
        #背景图片使用的是 SRGB 颜色空间，但 OpenGL 以线性颜色空间渲染，可能导致颜色混合变暗
        glEnable(GL_FRAMEBUFFER_SRGB)
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

        # # #坐标轴旋转回到世界坐标轴,注意旋转顺序必须倒过来因为旋转操作没有交换性
        # glRotatef(-self.rotation[2], 0, 0, 1)
        # glRotatef(-self.rotation[1], 0, 1, 0)
        # glRotatef(-self.rotation[0], 1, 0, 0)
        #平移回原点     
        #glTranslatef(-self.translation[0], -self.translation[1], -self.translation[2])
        glScalef(0.5, 0.5, 0.5)  # 坐标轴的缩放（固定大小）
        # 画坐标轴
        self.draw_axes()

##TODO处理新的点列和面列
    def init_mesh_data(self, unique_vertices, indices, is_blender=False):
        # 构造交错的顶点数据数组： [x, y, z, u, v, x, y, z, u, v, ...]
        self.vertex_data = []

        for pos, uv in unique_vertices:
            self.vertex_data.extend(pos.tolist())
            if is_blender:
                self.vertex_data.extend(uv.tolist())
            else:
                self.vertex_data.extend([uv[0].item(), 1 - uv[1].item()])  # y 轴翻转

        self.vertex_data = np.array(self.vertex_data, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)

        # 创建 VAO 和 VBO
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 启用混合以实现透明效果
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)  # 禁止写入深度缓冲区，防止背景当到物体前面
        # 绘制背景图片
        if self.background_image_path:
            #print("draw background")
            self.draw_background()
        glDepthMask(GL_TRUE)   # 允许写入深度缓冲区
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

        # glRotatef(self.rotation[0], 1, 0, 0)
        # glRotatef(self.rotation[1], 0, 1, 0)
        # glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(*self.scale)



        if self.diffuse_map is not None:
            # 初始化网格数据
            #TODO兼容性修改
            self.init_mesh_data(self.unique_vertices,self.unique_faces,is_blender=False)
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
        # else: 
        #     # 如果没有纹理，则绘制白色网格
        #     glColor3f(1.0, 1.0, 1.0)
        #     glBegin(GL_TRIANGLES)
        #     vertex_data = []
        #     for pos, uv in self.unique_vertices:
        #         vertex_data.extend(pos.tolist())
        #     for i in range(len(self.unique_faces)):
        #         index = self.unique_faces[i]
        #         vertex = vertex_data[index]
        #         glVertex3f(vertex[0][0], vertex[0][1], vertex[0][2])
        #     glEnd()

        self.draw_axes_with_camera()        
        # 禁用混合
        glDisable(GL_BLEND)

    def draw_quad(self):
        vertices = [
            # 位置          # 纹理坐标
            -1, -1, 0, 0.0, 0.0,
             1, -1, 0, 1.0, 0.0,
             1,  1, 0, 1.0, 1.0,
            -1,  1, 0, 0.0, 1.0
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
        """使用 OpenGL 纹理绘制背景"""
         # 绑定着色器程序
        glUseProgram(self.shader_program)

        # 绑定纹理
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.background_texture)
        glUniform1i(glGetUniformLocation(self.shader_program, "textureSampler"), 0)
        #GL_TRUE 表示 转置矩阵,但是不要用这个参数，我也不知道为什么用了会出错不如自己手动转置
        C =    np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ], dtype=np.float32)
        model = np.eye(4, dtype=np.float32)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = C
        projection = self.orthographic_projection(-1, 1, -1, 1, -10, 10)
        #projection = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 1, GL_FALSE, model.T)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 1, GL_FALSE, view.T)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 1, GL_FALSE, projection.T)
        # 绘制四边形
        self.draw_quad()

    def load_background_texture(self):
        """将背景图像转换为 OpenGL 纹理"""
        image = QImage(self.background_image_path)
        image = image.convertToFormat(QImage.Format_RGBA8888)
         # 获取图像数据并转换为bytes
        img_data = image.bits().asstring(image.sizeInBytes())
        width, height = image.width(), image.height()

        ptr = image.bits()
        ptr.setsize(image.byteCount())
        img_data = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))  # QImage 默认 RGBA

        #
        if self.camera_param is not None and self.distortion_enabled:
            print("去畸变")
            K = np.array(self.camera_param['K'], dtype=np.float32).reshape((3, 3))
            distCoeffs = np.array(self.camera_param['dist'], dtype=np.float32)

            # OpenCV 期望输入是 **BGR** 格式，先转换
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)

            # 进行去畸变
            undistorted_img = cv2.undistort(img_bgr, K, distCoeffs)

            # 转回 OpenGL 需要的 **RGBA**
            img_data = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGBA)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self.background_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.background_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)


    def set_distortion_enabled(self, enabled):
        self.distortion_enabled = enabled
        if self.background_image_path:
            self.load_background_texture()
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
            # 使用PyTorch3D加载OBJ文件
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            verts, faces, aux = load_obj(
                filename,
                load_textures=True,
                device=device,
                create_texture_atlas=True,
                texture_atlas_size=4,
            )
            # print(verts)
            # print(faces)
            # print(aux)
            # 转换顶点数据
            vertices = verts.cpu().numpy()
            
            # 转换面数据(注意OBJ索引从1开始，PyTorch3D会转换为0开始，之前我们的处理也是从0开始)
            # faces_idx = faces.verts_idx.cpu().numpy().tolist()
            v_idx = faces.verts_idx.cpu().numpy()
            vt_idx = faces.textures_idx.cpu().numpy()
            # 转换法线数据(如果有)
            normals = aux.normals.cpu().numpy() if aux.normals is not None else []
            # if aux.verts_uvs is None:
            #     QMessageBox.critical(None, "加载失败", "OBJ 文件未包含纹理坐标（vt），请检查模型导出设置。")
            #     QApplication.quit()  # 或者 sys.exit()
            # 转换纹理坐标(如果有)
            textures = aux.verts_uvs.cpu().numpy() if aux.verts_uvs is not None else []
            print(f"Number of vertices: {len(vertices.tolist())}")
            print(f"Number of faces: {len(v_idx.tolist())}")
            print(f"Number of tex: {len(textures.tolist())}")
            print(f"Number of normals: {len(normals.tolist())}")
            # 面索引：顶点索引和纹理索引

            # 构建唯一的 (v_idx, vt_idx) 映射
            #unique_vertices为新的点列（x,y,z,u,v)，indices为新的面列表（面数不变）
            unique_vertices = []
            vertex_map = {}   # key: (v, vt) → index
            indices = []
            for i in range(v_idx.shape[0]):
                for j in range(3):
                    vi = v_idx[i, j].item()
                    ti = vt_idx[i, j].item()
                    key = (vi, ti)
                    if key not in vertex_map:
                        pos = vertices[vi]
                        uv = textures[ti]
                        unique_vertices.append((pos, uv))
                        vertex_map[key] = len(unique_vertices) - 1
                    indices.append(vertex_map[key])

            #从obj得到mtl路径
            def get_mtllib_path(obj_path):
                with open(obj_path, 'r') as f:
                    for line in f:
                        if line.startswith('mtllib'):
                            return line.split()[1]
                return None
            #读取mtl文件实际上我们只需要贴图路径，当然也可以直接用pytorch3d读取的纹理图像但是懒得改了
            mtl_filename = get_mtllib_path(filename)
            if mtl_filename is None:
                print("No mtllib found in the OBJ file.")
            else:
                print(f"mtllib found: {mtl_filename}")
                obj_dir = os.path.dirname(os.path.abspath(filename))
                mtl_path = os.path.join(obj_dir, mtl_filename)
                mtl_materials = self.parseMTL(mtl_path)
                # 读取材质文件，假设只有一个，实际上如果我们把材质id考虑进去可以处理多材质的情况
                for mat_name, mat_data in aux.material_colors.items():
                    material = mtl_materials[mat_name]
                print(f"Material: {material}")
            

            # 更新所有的OpenGL小部件
            for widget in [self.glWidget1, self.glWidget2, self.glWidget3, 
                        self.glWidget4, self.glWidget5, self.glWidget6]:
                widget.set_mesh(unique_vertices, indices, material)
                
        except Exception as e:
            print(f"Error loading OBJ file with PyTorch3D: {e}")
 
    def read_intrinsics(self, yaml_path):
    # 打开YAML文件
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        
        # 读取所有name
        names_node = fs.getNode("names")
        names = []
        for i in range(names_node.size()):
            names.append(names_node.at(i).string())

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
            names.append(names_node.at(i).string())

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
#TODO:解析mtl文件
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