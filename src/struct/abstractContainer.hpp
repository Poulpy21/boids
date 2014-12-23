
#ifndef CONTAINER_H
#define CONTAINER_H

template <typename E> 
class AbstractContainer {
    static_assert(std::is_assignable<E&, E>(),        "E should be assignable !");
    static_assert(std::is_default_constructible<E>(), "E should be default constructible !");

    public:
        AbstractContainer() : _elements(0u), _size(0u), _data(0) {}
        AbstractContainer(const AbstractContainer<E> &other) : 
            _elements(other._elements), _size(other._size), _data(other._data) {}
        virtual ~AbstractContainer() {}
        
        E& operator[] (unsigned int k)       { return _data[k]; }
        E  operator[] (unsigned int k) const { return _data[k]; }

        unsigned int elements() const { return _elements; }
        unsigned int size()     const { return _size; }
        E* data() { return _data; }

        inline void insert(const E& e)    { _data[_elements] = e; _elements++; }
        inline void push_back(const E& e) { insert(e); }
        inline E pop_back() { _elements--; return _data[_elements]; }

        virtual void allocate(unsigned int minData)   = 0;
        //virtual void reallocate(unsigned int minData) = 0;
   
    protected:
        unsigned int _elements;
        unsigned int _size;
        E* _data;
};

#endif /* end of include guard: CONTAINER_H */
