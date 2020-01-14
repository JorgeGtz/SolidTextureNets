# render_cube.py
#
# Loads a solid texture (.npy file) and uses it to render a cube

import sys
import ctypes

from OpenGL import GL, GLU
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import sdl2
from sdl2 import video
import numpy as np

import glm
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at

windowheight = 800
windowwidth = 800
texture = './Trained/2020-01-10_brown016_exemplar_3D_2036/offline_ondemand_volume.npy'
vertex_offset = c_void_p(0 * sizeof(c_float))
normal_offset = c_void_p(3 * sizeof(c_float))
stride = 6 * sizeof(c_float)
lightVec = glm.vec3(-10, 5, 5)

def loadTexture(path):
    img = np.load(path)
    img = img.transpose([1,3,2,0])
    width, height , depth, channels = img.shape
    img_data = img.reshape(-1)

    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_BORDER)
    GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA, width, height, depth, 0,
        GL.GL_BGR, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_3D)
    return texture

def initialize():
    global shaderProgram
    global VAO
    global idxCnt
    global texUnitUniform
    global sampleTexture

    vertexStaticShader = shaders.compileShader("""
#version 330 core
layout (location=0) in vec3 position;
layout (location=1) in vec3 aNormal;

uniform mat4 rot;
uniform mat4 translation;
uniform mat4 projection;
uniform mat4 view;

out vec3 theCoords;
out vec3 theNormal;

void main()
{
    vec4 world = translation*rot*vec4((1*position), 1);
    gl_Position = projection*view*world;
    theCoords = 0.5*(vec3(1,1,1)+1*position);
    theNormal = aNormal;
}
""", GL.GL_VERTEX_SHADER)

    fragmentShader = shaders.compileShader("""
#version 330 core
uniform sampler3D texUnit;

in vec3 theCoords;
in vec3 theNormal;

uniform vec3 lightPos;

out vec4 outputColour;

float ambientStrenght;
float diff;
vec3 lightColor;
vec3 ambient;
vec3 norm;
vec3 lightDir;
vec3 diffuse;

void main()
{
    // ambient
    lightColor = vec3(1, 1, 1);
    ambientStrenght = 1.0;
    ambient = ambientStrenght * lightColor;

    // diffusse
    norm = normalize(theNormal);
    lightDir = normalize(lightPos - theCoords);
    diff = max(dot(norm, lightDir), 0.0);
    diffuse = diff * lightColor;

    // out
    outputColour = vec4(diffuse+ambient, 1.0)*texture(texUnit, theCoords);
}
""", GL.GL_FRAGMENT_SHADER)


    vertexData = np.array([
     1.0, -1.0, -1.0, 0,-1,0,
     1.0, -1.0,  1.0, 0,-1,0,
    -1.0, -1.0,  1.0, 0,-1,0,
    -1.0, -1.0, -1.0, 0,-1,0,
     1.0,  1.0, -1.0, 0,1,0,
     1.0,  1.0,  1.0, 0,1,0,
    -1.0,  1.0,  1.0, 0,1,0,
    -1.0,  1.0, -1.0, 0,1,0,
     1.0, -1.0, -1.0, 1,0,0,
     1.0, -1.0,  1.0, 1,0,0,
     1.0,  1.0, -1.0, 1,0,0,
     1.0,  1.0,  1.0, 1,0,0,
     1.0, -1.0, -1.0, 0,0,-1,
    -1.0, -1.0, -1.0, 0,0,-1,
     1.0,  1.0, -1.0, 0,0,-1,
    -1.0,  1.0, -1.0, 0,0,-1,
    -1.0, -1.0,  1.0, -1,0,0,
    -1.0, -1.0, -1.0, -1,0,0,
    -1.0,  1.0,  1.0, -1,0,0,
    -1.0,  1.0, -1.0, -1,0,0,
     1.0, -1.0,  1.0, 0,0,1,
    -1.0, -1.0,  1.0, 0,0,1,
     1.0,  1.0,  1.0, 0,0,1,
    -1.0,  1.0,  1.0, 0,0,1,
    ],dtype=np.float32)
    faceData = np.array([
        1,3,0,
        7,5,4,
        22,21,20,
        10,9,8,
        12,15,14,
        16,19,17,
        1,2,3,
        7,6,5,
        22,23,21,
        10,11,9,
        12,13,15,
        16,18,19,
    ],dtype=np.uint32)
    idxCnt = faceData.shape[0]

    GL.glEnable(GL.GL_DEPTH_TEST);
    GL.glDepthFunc(GL.GL_LEQUAL);

    shaderProgram = shaders.compileProgram(vertexStaticShader, fragmentShader)

    GL.glDeleteShader(vertexStaticShader);
    GL.glDeleteShader(fragmentShader);

    VAO = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(VAO)

    # Need VBO for triangle vertices  coordinates
    VBO = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData,
        GL.GL_STATIC_DRAW)

    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, vertex_offset)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, normal_offset)
    GL.glEnableVertexAttribArray(1)

    IBO = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, IBO)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, faceData.nbytes, faceData,
        GL.GL_STATIC_DRAW)

    # Finished
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    # load texture and assign texture unit for shaders
    sampleTexture = loadTexture(texture)
    texUnitUniform = GL.glGetUniformLocation(shaderProgram, 'texUnit')

def render():
    global sampleTexture
    global shaderProgram
    global texUnitUniform
    global VAO
    global idxCnt
    global rot
    global projection
    global translation
    global view
    global lightVec
    GL.glClearColor(1, 1, 1, 1)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(shaderProgram)

    try:
        rotUniform = GL.glGetUniformLocation(shaderProgram, 'rot')
        GL.glUniformMatrix4fv(rotUniform, 1, False, glm.value_ptr(rot))
        projectionLoc = GL.glGetUniformLocation(shaderProgram, 'projection')
        GL.glUniformMatrix4fv(projectionLoc, 1, False, glm.value_ptr(projection))
        translationLoc = GL.glGetUniformLocation(shaderProgram, 'translation')
        GL.glUniformMatrix4fv(translationLoc, 1, False, glm.value_ptr(translation))
        viewLoc = GL.glGetUniformLocation(shaderProgram, 'view')
        GL.glUniformMatrix4fv(viewLoc, 1, False, glm.value_ptr(view))
        lightLoc = GL.glGetUniformLocation(shaderProgram, 'lightPos')
        GL.glUniform3fv(lightLoc, 1, glm.value_ptr(lightVec))
        # Activate texture
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, sampleTexture)
        GL.glUniform1i(texUnitUniform, 0)
        # Activate array
        GL.glBindVertexArray(VAO)
        GL.glDrawElements(GL.GL_TRIANGLES, idxCnt, GL.GL_UNSIGNED_INT, None)


    finally:
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


def run():
    global shaderProgram
    global rot
    global projection
    global translation
    global view
    global lightVec
    if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
        print(sdl2.SDL_GetError())
        return -1

    window = sdl2.SDL_CreateWindow(b"OpenGL demo",
                                   sdl2.SDL_WINDOWPOS_UNDEFINED,
                                   sdl2.SDL_WINDOWPOS_UNDEFINED, windowwidth, windowheight,
                                   sdl2.SDL_WINDOW_OPENGL)
    if not window:
        print(sdl2.SDL_GetError())
        return -1

    video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
    video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MINOR_VERSION, 3)
    video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_PROFILE_MASK,
        video.SDL_GL_CONTEXT_PROFILE_CORE)
    context = sdl2.SDL_GL_CreateContext(window)

    # Setup GL shaders, data, etc.
    initialize()
    event = sdl2.SDL_Event()
    running = True

    # Initial position
    animated = 1
    init_t = glm.vec3(0.0,0.0,-4.0)
    rot = glm.mat4(1.0)
    rot = glm.rotate(rot,glm.radians(45),glm.vec3(0.0,1.0,0.0))
    rot = glm.rotate(rot,glm.radians(20),glm.vec3(1.0,0.0,0.0))
    rot = glm.rotate(rot,glm.radians(20),glm.vec3(0.0,0.0,1.0))
    step = 0.5
    translation = glm.mat4(1.0)
    translation = glm.translate(translation,init_t)
    projection = glm.perspective(glm.radians(45.0), windowwidth / windowheight, 1.0, 10.0)

    cameraPos   = glm.vec3(0.0, 0.0, 1.0)
    cameraFront = glm.vec3(0.0, 0.0, -1.0)
    cameraUp    = glm.vec3(0.0, 1.0, 0.0)
    view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

    while running:
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                running = False
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_ESCAPE):
                running = False

            # Animation
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_s):
                if animated == 1:
                    animated = 0
                else:
                    animated = 1

            # Rotation
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_RIGHT):
                rot = glm.rotate(rot,glm.radians(step),glm.vec3(0.0,0.0,1.0))
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_LEFT):
                rot = glm.rotate(rot,glm.radians(-step),glm.vec3(0.0,0.0,1.0))
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_q):
                rot = glm.rotate(rot,glm.radians(step),glm.vec3(0.0,1.0,0.0))
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_a):
                rot = glm.rotate(rot,glm.radians(-step),glm.vec3(0.0,1.0,0.0))
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_UP):
                rot = glm.rotate(rot,glm.radians(step),glm.vec3(1.0,0.0,0.0))
            elif (event.type == sdl2.SDL_KEYDOWN and
                event.key.keysym.sym == sdl2.SDLK_DOWN):
                rot = glm.rotate(rot,glm.radians(-step),glm.vec3(1.0,0.0,0.0))

        render()

        if animated==1:
            rot = glm.rotate(rot,glm.radians(step),glm.vec3(0.0,1.0,0.0))

        sdl2.SDL_GL_SwapWindow(window)
        sdl2.SDL_Delay(10)

    sdl2.SDL_GL_DeleteContext(context)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()
    return 0

if __name__ == "__main__":
    sys.exit(run())
