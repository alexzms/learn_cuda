//
// Created by alexzms on 2023/10/1.
//

#ifndef LEARN_OPENGL_CLION_SHADER_H
#define LEARN_OPENGL_CLION_SHADER_H

#include <glm/gtc/type_ptr.hpp>
#include "glad/glad.h"

#include "string"
#include "iostream"
#include "fstream"
#include "sstream"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Shader{
public:
    // shader ID
    unsigned int ID;
    bool functional = false;

    Shader(const char* vertex_path, const char* fragment_path);

    // activate program
    void use() const;
    // uniform value set
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    void setVec3(const std::string &name, glm::vec3 vec) const;
    void setMat4(const std::string &name, glm::mat4 mat) const;
};

Shader::Shader(const char* vertex_path, const char* fragment_path) {
    bool set_functional = true;
    std::string vertex_code;
    std::string fragment_code;

    std::ifstream vertex_file;
    std::ifstream fragment_file;
    vertex_file.exceptions(std::ifstream::failbit | std::ifstream ::badbit);
    fragment_file.exceptions(std::ifstream::failbit | std::ifstream ::badbit);

    try {
        vertex_file.open(vertex_path);
        fragment_file.open(fragment_path);

        std::stringstream vertex_stream, fragment_stream;
        vertex_stream << vertex_file.rdbuf();
        fragment_stream << fragment_file.rdbuf();

        vertex_file.close();
        fragment_file.close();

        vertex_code = vertex_stream.str();
        fragment_code = fragment_stream.str();
    } catch (std::ifstream ::failure &e) {
        std::cout << "ERROR::Shader::File not read successfully" << std::endl;
    }

    const char* vertex_shader_code = vertex_code.c_str();
    const char* fragment_shader_code = fragment_code.c_str();

    unsigned int vertex_shader, fragment_shader;
    int success;
    char info_log[512];

    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_code, nullptr);
    glCompileShader(vertex_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        set_functional = false;
        glGetShaderInfoLog(vertex_shader, 512, nullptr, info_log);
        std::cout << "ERROR::Shader::Vertex shader compile failed\n" << info_log << std::endl;
    }

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_code, nullptr);
    glCompileShader(fragment_shader);
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        set_functional = false;
        glGetShaderInfoLog(fragment_shader, 512, nullptr, info_log);
        std::cout << "ERROR::Shader::Fragment shader compile failed\n" << info_log << std::endl;
    }

    Shader::ID = glCreateProgram();
    glAttachShader(ID, vertex_shader);
    glAttachShader(ID, fragment_shader);
    glLinkProgram(ID);

    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if (!success) {
        set_functional = false;
        glGetProgramInfoLog(ID, 512, nullptr, info_log);
        std::cout << "ERROR::Shader::Shader program link failed\n" << info_log << std::endl;
    }

    Shader::functional = set_functional;
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
}

void Shader::use() const{
    glUseProgram(Shader::ID);
}

void Shader::setBool(const std::string &name, bool value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int) value);
}

void Shader::setInt(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setFloat(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setVec3(const std::string &name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void Shader::setVec3(const std::string &name, glm::vec3 vec) const {
    glUniform3f(glGetUniformLocation(ID, name.c_str()), vec.x, vec.y, vec.z);
}

void Shader::setMat4(const std::string &name, glm::mat4 mat) const {
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

#endif //LEARN_OPENGL_CLION_SHADER_H
