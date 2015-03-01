
#ifndef CONTAINER_H
#define CONTAINER_H

#include "headers.hpp"
#include "localized.hpp"
#include <type_traits>

#ifdef GUI_ENABLED
template <typename E> 
class AbstractContainer {
#else
template <typename E> 
class AbstractContainer : public Renderable {
#endif

    static_assert(std::is_assignable<E&, E>(),        "E should be assignable !");
    static_assert(std::is_default_constructible<E>(), "E should be default constructible !");

    public:
        AbstractContainer();
        AbstractContainer(const AbstractContainer<E> &other) : 
            _elements(other._elements), _size(other._size), _data(other._data) {
                for (unsigned int i = 0; i < 3; i++) {
                    _color[i] = other._color[i];
                }
            }
        virtual ~AbstractContainer() {}
        
        E& operator[] (unsigned int k)       { return _data[k]; updateGLBuffer(); }
        E  operator[] (unsigned int k) const { return _data[k]; }

        unsigned int elements() const { return _elements; }
        unsigned int size()     const { return _size; }
        float fillRate()        const { return static_cast<float>(_elements) / static_cast<float>(_size); };
        E* data() { return _data; }

        void insert(const E& e)    { _data[_elements] = e; _elements++; updateGLBuffer(); }
        void remove(unsigned int elementId) {
            assert(elementId < _elements);
            memcpy(_data + elementId, _data + _elements - 1, sizeof(E));
            _elements --;
            updateGLBuffer();
        }

        void push_back(const E& e) { insert(e); }
        E pop_back() { _elements--; return _data[_elements]; updateGLBuffer(); }

        virtual void allocate(unsigned int minData)   = 0;
        virtual void reallocate(unsigned int minData) = 0;
   
    protected:
        unsigned int _elements;
        unsigned int _size;
        E* _data;

#ifdef GUI_ENABLED
    public:
        template <typename A>
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4) override;
        
    protected:
        void makeProgram();
        void makeVAO();
       
        void updateGLBuffer() { 
            _boidBufferUpdate = true;
        }
        
        static Program *_program;
        static std::map<std::string, int> _uniformLocations;

        unsigned int _VAO, _VBO;
        bool _boidBufferUpdate;
        float _color[3];
#endif

};


template <typename E> 
AbstractContainer<E>::AbstractContainer() : _elements(0u), _size(0u), _data(0) 
#ifdef GUI_ENABLED
    ,_VAO(0u), _VBO(0u), _boidBufferUpdate(false) {
        for (unsigned int i = 0; i < 3; i++) {
            _color[i] = Random::randf();
        }
    }
#else
{}
#endif

#ifdef GUI_ENABLED

template <typename E> 
Program* AbstractContainer<E>::_program = nullptr;

template <typename E> 
std::map<std::string, int> AbstractContainer<E>::_uniformLocations;
        
template <typename E> 
template <typename A>
void AbstractContainer<E>::drawDownwards(const float *currentTransformationMatrix) {
    static_assert(std::is_base_of<Localized<3u,A>,E>(), "E should be localized in 3D !");
    
    makeProgram();
    makeVAO();

    if(this->elements() == 0)
        return;

    if (_boidBufferUpdate) {
        glBindBuffer(GL_ARRAY_BUFFER, _VBO);

        float *positionBuffer = new float[this->elements()*3u];
        for (unsigned int i = 0; i < this->elements(); i++) {
            Vec3<A> newPos = _data[i].position();
            positionBuffer[3*i + 0] = static_cast<float>(newPos.x);
            positionBuffer[3*i + 1] = static_cast<float>(newPos.y);
            positionBuffer[3*i + 2] = static_cast<float>(newPos.z);
        }

        glBufferSubData(GL_ARRAY_BUFFER, 0, 3u*this->elements()*sizeof(float), positionBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        delete [] positionBuffer;
        _boidBufferUpdate = false;
    }

	_program->use();

    glUniform3fv( _uniformLocations["inColor"], 1, _color);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);

	glBindVertexArray(_VAO);
        glDrawArrays(GL_POINTS, 0, this->elements());
	glBindVertexArray(0);

	glUseProgram(0);
}
        
template <typename E> 
void AbstractContainer<E>::makeProgram() {
    if(_program == nullptr) {
        _program = new Program("Boids");

        _program->bindAttribLocations("0", "vertexPosition");
        _program->bindFragDataLocation(0, "outColor");
        _program->bindUniformBufferLocations("0", "projectionView");

        _program->attachShader(Shader(Globals::shaderFolder + "/boids/boids_vs.glsl", GL_VERTEX_SHADER));
        _program->attachShader(Shader(Globals::shaderFolder + "/boids/boids_fs.glsl", GL_FRAGMENT_SHADER));
		
        _program->link();
        
        _uniformLocations =  _program->getUniformLocationsMap("inColor", true);
    }
}

template <typename E> 
void AbstractContainer<E>::makeVAO() {
    if(!glIsBuffer(_VBO)) {
        
        glGenVertexArrays(1, &_VAO);
        glBindVertexArray(_VAO);
        
        glGenBuffers(1, &_VBO);
        glBindBuffer(GL_ARRAY_BUFFER, _VBO);
        glBufferData(GL_ARRAY_BUFFER, 3*this->size()*sizeof(float), 0, GL_DYNAMIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);

        updateGLBuffer();
    }
}
    
#endif /* GUI_ENABLED */

#endif /* end of include guard: CONTAINER_H */
